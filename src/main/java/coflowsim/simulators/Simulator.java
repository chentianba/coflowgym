package coflowsim.simulators;

import java.util.HashMap;
import java.util.Random;
import java.util.Vector;
import java.util.ArrayList;

import coflowsim.datastructures.Flow;
import coflowsim.datastructures.Job;
import coflowsim.datastructures.JobCollection;
import coflowsim.traceproducers.TraceProducer;
import coflowsim.utils.Constants;
import coflowsim.utils.Constants.SHARING_ALGO;
import coflowsim.utils.Utils;

/**
 * Abstract class for all schedulers.
 * This class implements the core event loop of the simulator and exposes methods to respond to
 * events, which must be overwritten by different schedulers for specific behavior.
 */
public abstract class Simulator {

  public static int NUM_RACKS = 150;
  public static int MACHINES_PER_RACK = 20;

  protected JobCollection jobs;

  protected Vector<Flow>[] flowsInRacks;
  protected HashMap<String, Job> activeJobs;

  protected SHARING_ALGO sharingAlgo;

  protected boolean isOffline;
  protected boolean considerDeadline;

  double deadlineMultRandomFactor;

  protected long CURRENT_TIME = 0;
  protected int stepJobsAdded = 0;

  private int numActiveTasks = 0;

  /* Defined by chentb
   * used in function Step()
   */
  private int curStepJob = 0;
//   private int curStepTime = 0;

  /**
   * Constructor for Simulator.
   * 
   * @param sharingAlgo
   *          {@link coflowsim.utils.Constants.SHARING_ALGO} to be used.
   * @param traceProducer
   *          {@link coflowsim.traceproducers.TraceProducer} to simulate.
   * @param offline
   *          whether all coflows/jobs should be forced start together irrespective of their
   *          arrival times.
   * @param considerDeadline
   *          whether or not to ignore deadlines.
   * @param deadlineMultRandomFactor
   *          multiplier used to generate deadlines.
   */
  public Simulator(
      SHARING_ALGO sharingAlgo,
      TraceProducer traceProducer,
      boolean offline,
      boolean considerDeadline,
      double deadlineMultRandomFactor) {

    NUM_RACKS = traceProducer.getNumRacks();
    MACHINES_PER_RACK = traceProducer.getMachinesPerRack();

    this.sharingAlgo = sharingAlgo;
    this.isOffline = offline;
    this.considerDeadline = considerDeadline;
    this.deadlineMultRandomFactor = deadlineMultRandomFactor;

    initialize(traceProducer);
  }

  /**
   * Initialize the simulator and relevant data structures.
   * 
   * @param traceProducer
   *          {@link coflowsim.traceproducers.TraceProducer} to simulate.
   */
  @SuppressWarnings("unchecked")
  protected void initialize(TraceProducer traceProducer) {
    this.jobs = traceProducer.jobs;
    this.jobs.sortByStartTime();

    this.flowsInRacks = (Vector<Flow>[]) new Vector[NUM_RACKS];
    for (int i = 0; i < NUM_RACKS; i++) {
      flowsInRacks[i] = new Vector<Flow>();
    }

    this.activeJobs = new HashMap<String, Job>();

    // Merge tasks
    mergeTasksByRack();
  }

  /**
   * Merges all tasks in the same rack to a single one to form a non-blocking switch.
   * <p>
   * This is here for historical purposes, because actual production clusters at Facebook and
   * Microsoft have core-rack over-subscription; the non-blocking starts from racks and not from
   * machines.
   */
  private void mergeTasksByRack() {
    Random deadlineGen = new Random(17);
    for (Job j : jobs) {
      if (!ignoreThisJob(j)) {
        double deadlineMult = 1.0 + deadlineGen.nextDouble() * deadlineMultRandomFactor;
        j.arrangeTasks(NUM_RACKS, MACHINES_PER_RACK, deadlineMult);
      }
    }
  }

  /**
   * Whether or not to ignore a specific job or set of jobs.
   * Useful for debugging.
   * 
   * @param j
   *          job to decide upon
   * @return decision
   */
  protected boolean ignoreThisJob(Job j) {
    if (j.numMappers == 0) {
      return true;
    }
    return false;
  }

  /**
   * Determine whether to admit a job or not.
   * <p>
   * Admits all jobs by default. Must be overwritten by schedulers supporting admission control
   * (e.g., {@link coflowsim.simulators.CoflowSimulator}).
   * 
   * @param j
   *          job to decide upon
   * @return admission decision
   */
  protected boolean admitThisJob(Job j) {
    return true;
  }

  /**
   * Event loop of the simulator that proceed epoch by epoch.
   * <ul>
   * <li>Admit/reject individual jobs/coflows at the beginning of each epoch and update relevant
   * data structures using {@link #uponJobAdmission(Job)}.
   * <li>Once admission process if over, perform common tasks for all the admitted ones using
   * {@link #afterJobAdmission(long)}.
   * <li>Simulate the time steps in each epoch, where each time step is as long as it takes to
   * transfer 1MB through each link.
   * <li>In each time step take appropriate scheduling decision using {@link #onSchedule(long)}.
   * <li>If any job/coflow has completed, update relavant data structures using
   * {@link #afterJobDeparture(long)}.
   * <li>Repeat.
   * </ul>
   * 
   * @param EPOCH_IN_MILLIS
   *          simulator epoch length in milliseconds
   */
  public void simulate(int EPOCH_IN_MILLIS) {
    // EPOCH_IN_MILLIS /= 10;
    Utils.log("EPOCH_IN_MILLIS:"+EPOCH_IN_MILLIS);
    int curJob = 0;
    int TOTAL_JOBS = jobs.size();

    for (CURRENT_TIME = 0; CURRENT_TIME < Constants.SIMULATION_ENDTIME_MILLIS
        && (curJob < TOTAL_JOBS || numActiveTasks > 0); CURRENT_TIME += EPOCH_IN_MILLIS) {
      Utils.log("\n=====> foreach current_time = "+CURRENT_TIME);

      int jobsAdded = 0;

      // Queue up all tasks in all jobs within SIMULATION_TIMESTEP
      for (; curJob < TOTAL_JOBS; curJob++) {
        Job j = jobs.elementAt(curJob);
        if (j.actualStartTime > CURRENT_TIME + EPOCH_IN_MILLIS) {
          break;
        }

        if (ignoreThisJob(j)) {
          continue;
        }

        if (!admitThisJob(j)) {
          System.err.println("SKIPPING " + j);
          continue;
        }

        // One job added
        j.wasAdmitted = true;
        jobsAdded++;
        uponJobAdmission(j);
      }

      // Stuff to do on new job arrival
      if (jobsAdded > 0) {
        afterJobAdmission(CURRENT_TIME);
      }

    //   System.out.println("jobsAdded = "+jobsAdded);

      Utils.log("Before Scheduling!");
      for (Job t : activeJobs.values()) {
        Utils.log(t.toString());
      }
      for (long i = 0; i < EPOCH_IN_MILLIS; i += Constants.SIMULATION_QUANTA) {

        int numActiveJobs = activeJobs.size();

        if (numActiveJobs == 0) {
        //   System.out.println("numActiveJobs==0 i="+i);
          break;
        }

        long curTime = CURRENT_TIME + i;
        onSchedule(curTime);

        // Print progress
        if (curTime % Constants.SIMULATION_SECOND_MILLIS == 0) {
          System.err.printf("Timestep %6d: Running: %3d Started: %5d\n",
              (curTime / Constants.SIMULATION_SECOND_MILLIS), numActiveJobs, curJob);
        }

        // Stuff after job departures
        if (numActiveJobs > activeJobs.size()) {
          afterJobDeparture(curTime);
        }
        // System.out.println("i = "+i+" numActiveJobs="+numActiveJobs);
      }

      Utils.log("After Scheduling!");
      for (Job t : activeJobs.values()) {
        Utils.log(t.toString());
      }
    }
  }

    public boolean prepareActiveJobs(int EPOCH_IN_MILLIS) {
        int TOTAL_JOBS = jobs.size();
        boolean done = false;
        String obs = "";

        if (curStepJob >= TOTAL_JOBS && numActiveTasks <= 0) {
            done = true;
            return true;
        }
        if (CURRENT_TIME >= Constants.SIMULATION_ENDTIME_MILLIS) {
            done = true;
            return true;
        }

        Utils.log("\n=====> current_step_time = "+CURRENT_TIME);

        stepJobsAdded = 0;
        // ArrayList<Job> obsJobs = new ArrayList<Job>();

        // Queue up all tasks in all jobs within SIMULATION_TIMESTEP
        for (; curStepJob < TOTAL_JOBS; curStepJob++) {
            Job j = jobs.elementAt(curStepJob);
            if (j.actualStartTime > CURRENT_TIME + EPOCH_IN_MILLIS) {
                break;
            }

            if (ignoreThisJob(j)) {
                continue;
            }

            if (!admitThisJob(j)) {
                System.err.println("SKIPPING " + j);
                continue;
            }

            // One job added
            j.wasAdmitted = true;
            stepJobsAdded++;
            uponJobAdmission(j);
        }

        // for (Job j : activeJobs.values()) {
        //     // obsJobs.add(j);
        // }
        // System.out.println("activeJobs size: "+activeJobs.values().size());
        return false;
    }

    /**
     * Given a timestep and return "observation, reward, done, info".
     * 
     * @param STEP_IN_MILLIS
     *      timestep toward a step.
     * @return
     *      whether this trace is completed.
     */
    public String step(int STEP_IN_MILLIS) {
        int EPOCH_IN_MILLIS = STEP_IN_MILLIS; // / 10;
        String completed = "";

        // record jobs before scheduling
        ArrayList<Job> beforeJobs = new ArrayList<Job>();
        Utils.log("Before Scheduling!");
        for (Job t : activeJobs.values()) {
            Utils.log(t.toString());
            beforeJobs.add(t);
        }
        Utils.log("here");

        // Stuff to do on new job arrival
        if (stepJobsAdded > 0) {
            afterJobAdmission(CURRENT_TIME);
        }

        for (long i = 0; i < EPOCH_IN_MILLIS; i += Constants.SIMULATION_QUANTA) {

            int numActiveJobs = activeJobs.size();

            if (numActiveJobs == 0) {
            //   System.out.println("numActiveJobs==0 i="+i);
                break;
            }

            long curTime = CURRENT_TIME + i;
            onSchedule(curTime);

            // Print progress
            if (curTime % Constants.SIMULATION_SECOND_MILLIS == 0) {
                // System.err.printf("Timestep %6d: Running: %3d Started: %5d\n", (curTime / Constants.SIMULATION_SECOND_MILLIS), numActiveJobs, curStepJob);
            }

            // Stuff after job departures
            if (numActiveJobs > activeJobs.size()) {
                afterJobDeparture(curTime);
            }
            // System.out.println("i = "+i+" numActiveJobs="+numActiveJobs);
        }

        Utils.log("After Scheduling!");
        for (Job t : activeJobs.values()) {
            Utils.log(t.toString());
        }

        // create completed coflow info
        completed += "completed: [";
        for (Job j : beforeJobs) {
            if (!activeJobs.values().contains(j)) {
                completed += "(" + (j.jobID+", "+j.numMappers*j.numReducers+", "+j.totalShuffleBytes+", "+(j.getSimulatedDuration())) + "), "; // id, width, already bytes, duration time
            }
        }
        completed += "]";

        CURRENT_TIME += EPOCH_IN_MILLIS;

        return completed;
    }

    /**
     * Get a snapshot of coflow.
     * @return
     *      a string in JSON format about (id, width, sent bytes, duration time)
     *   sent bytes: unit is B(byte)
     *   duration time: unit is milli second, a second = 1024 milli second
     */
    public String getObservation(int STEP_IN_MILLIS) {
        String obs = "";
        Utils.log("Observation:");
        obs += "Observation: [";
        for (Job j : activeJobs.values()) {
            // Utils.log(j.toString());
            obs += "(" + (j.jobID+", "+j.numMappers*j.numReducers+", "+j.shuffleBytesCompleted+", "+(CURRENT_TIME+STEP_IN_MILLIS-j.simulatedStartTime)) + "), "; // id, width, already bytes, duration time
        }
        obs += "]";

        return obs;
    }

  /**
   * Stuff to do before each epoch for each job that has been admitted.
   * 
   * @param j
   *          job that has been admitted
   */
  protected abstract void uponJobAdmission(Job j);

  /**
   * Stuff to do before each epoch if one or more jobs have been admitted.
   * 
   * @param curTime
   *          current time in milliseconds
   */
  protected void afterJobAdmission(long curTime) {
  }

  /**
   * Scheduling actions to make at every simulation time step.
   * 
   * @param curTime
   *          current time in milliseconds
   */
  protected abstract void onSchedule(long curTime);

  /**
   * Stuff to do after a simulation time step if one or more jobs have completed.
   * 
   * @param curTime
   *          current time in milliseconds
   */
  protected void afterJobDeparture(long curTime) {
  }

  /**
   * Calculates and prints summary statistics of the simulation.
   * 
   * @param doPrint
   *          print if true
   * @return the total coflow completion time
   */
  public String printStats(boolean doPrint) {
    double sumDur = 0.0;
    int admitCount = 0;
    int ignoreCount = 0;
    int metDeadlineCount = 0;
    String resStats = "";

    for (Job j : jobs) {
      if (ignoreThisJob(j)) {
        continue;
      }

      double jDur = j.getSimulatedDuration();
      sumDur += jDur;

      if (j.wasAdmitted) {
        admitCount++;
      } else {
        ignoreCount++;
      }

      boolean metDeadline = false;
      if (jDur - j.deadlineDuration < 100
          || ((jDur / 8.0) / (j.deadlineDuration * 128.0 / 1000.0)) - 1.0 < 0.03) {
        metDeadline = true;
      }
      if (j.wasAdmitted && metDeadline) {
        metDeadlineCount++;
      }

      if (doPrint) {
        String res = (j.jobName + " " + j.simulatedStartTime + " " + j.simulatedFinishTime
            + " " + j.numMappers + " " + j.numReducers + " " + j.totalShuffleBytes + " "
            + j.maxShuffleBytes + " " + jDur + " " + Math.round(j.deadlineDuration) + " "
            + j.simulatedShuffleIndividualSums);
        resStats += (res + "\n");
        // System.out.println(res);
      }
    }

    if (true){//doPrint) {
      resStats += sumDur;
    //   System.out.println(sumDur);
      if (considerDeadline) {
        System.out.println(metDeadlineCount + "/" + admitCount + " " + ignoreCount);
      }
    }

    return resStats;
  }

  public String getCoflowInfo() {
    String resStats = "";
    return resStats;
  }

  public void incNumActiveTasks() {
    numActiveTasks++;
  }

  public void decNumActiveTasks() {
    numActiveTasks--;
  }

  /**
   * Remove completed job from relevant data structures.
   * 
   * @param j
   *          job to remove
   */
  protected abstract void removeDeadJob(Job j);

  public boolean setThreshold(double[] thresholds) {
      return false;
  }

  public String getMLFQInfo() {
      return "";
  }
}
