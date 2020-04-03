package coflowsim;

import java.util.*;

import coflowsim.simulators.CoflowSimulator;
import coflowsim.simulators.CoflowSimulatorDark;
import coflowsim.simulators.FlowSimulator;
import coflowsim.simulators.Simulator;
import coflowsim.traceproducers.CoflowBenchmarkTraceProducer;
import coflowsim.traceproducers.CustomTraceProducer;
import coflowsim.traceproducers.JobClassDescription;
import coflowsim.traceproducers.TraceProducer;
import coflowsim.utils.Constants;
import coflowsim.utils.Constants.SHARING_ALGO;

public class CoflowGym {
    private Simulator simulator;
    private String[] input;

    public static int MAX_COFLOW = 10;

    public CoflowGym(String[] args) {
        input = args;
        this.initializeSimulator();
        assert this.simulator.getClass().equals(CoflowSimulatorDark.class);
    }

    /**
     * format "|=|=|" is 
     * activeJobs -> epoch scheduling -> activeJobs -> epoch scheduling 
     * @return
     *      return the observation
     */
    public String toOneStep(double[] thresholds) {
        // System.out.println("thresholds: "+Arrays.toString(thresholds));
        String res = "";
        int TIMESATMP = 10 * Constants.SIMULATION_SECOND_MILLIS;
        boolean done;
        String obs, completed;
        this.takeAction(thresholds);
        completed = this.simulator.step(TIMESATMP);
        done = this.simulator.prepareActiveJobs(TIMESATMP);
        obs = this.simulator.getObservation(TIMESATMP);
        res += ("{\"observation\":\""+obs+"\",");
        res += ("\"completed\":\""+completed+"\",");
        res += ("\"done\":"+done+"}");

        return res;
    }

    public String reset() {
        String obs;
        int TIMESATMP = 10 * Constants.SIMULATION_SECOND_MILLIS;
        this.initializeSimulator();
        this.simulator.prepareActiveJobs(TIMESATMP);
        obs = this.simulator.getObservation(TIMESATMP);
        return obs;
    }

    public String printStats() {
        return this.simulator.printStats(false);
        // System.out.println(this.simulator instanceof CoflowSimulatorDark);
    }

    public void takeAction(double[] thresholds) {
        boolean flag = false;
        flag = this.simulator.setThreshold(thresholds);
        if (!flag) {
            System.err.println("Action doesn't take effect!");
        }
    }

    public void initializeSimulator() {
        String[] args = this.input;       
        int curArg = 0;
    
        SHARING_ALGO sharingAlgo = SHARING_ALGO.FAIR;
        if (args.length > curArg) {
          String UPPER_ARG = args[curArg++].toUpperCase();
        //   System.out.println("sharingAlgo(argument): "+UPPER_ARG);
    
          if (UPPER_ARG.contains("FAIR")) {
            sharingAlgo = SHARING_ALGO.FAIR;
          } else if (UPPER_ARG.contains("PFP")) {
            sharingAlgo = SHARING_ALGO.PFP;
          } else if (UPPER_ARG.contains("FIFO")) {
            sharingAlgo = SHARING_ALGO.FIFO;
          } else if (UPPER_ARG.contains("SCF") || UPPER_ARG.contains("SJF")) {
            sharingAlgo = SHARING_ALGO.SCF;
          } else if (UPPER_ARG.contains("NCF") || UPPER_ARG.contains("NJF")) {
            sharingAlgo = SHARING_ALGO.NCF;
          } else if (UPPER_ARG.contains("LCF") || UPPER_ARG.contains("LJF")) {
            sharingAlgo = SHARING_ALGO.LCF;
          } else if (UPPER_ARG.contains("SEBF")) {
            sharingAlgo = SHARING_ALGO.SEBF;
          } else if (UPPER_ARG.contains("DARK")) {
            sharingAlgo = SHARING_ALGO.DARK;
          } else {
            System.err.println("Unsupported or Wrong Sharing Algorithm");
            System.exit(1);
          }
        }
    
        boolean isOffline = false;
        int simulationTimestep = 10 * Constants.SIMULATION_SECOND_MILLIS;
        if (isOffline) {
          simulationTimestep = Constants.SIMULATION_ENDTIME_MILLIS;
        }
    
        boolean considerDeadline = false;
        double deadlineMultRandomFactor = 1;
        if (considerDeadline && args.length > curArg) {
          deadlineMultRandomFactor = Double.parseDouble(args[curArg++]);
        }
    
        // Create TraceProducer
        TraceProducer traceProducer = null;
    
        int numRacks = 100;
        int numJobs = 10;
        int randomSeed = 13;
        JobClassDescription[] jobClassDescs = new JobClassDescription[] {
            new JobClassDescription(1, 5, 1, 10),
            new JobClassDescription(1, 5, 10, 1000),
            new JobClassDescription(5, numRacks, 1, 10),
            new JobClassDescription(5, numRacks, 10, 1000) };
        double[] fracsOfClasses = new double[] {
            41,
            29,
            9,
            21 };
    
        traceProducer = new CustomTraceProducer(numRacks, numJobs, jobClassDescs, fracsOfClasses,
            randomSeed);
    
        if (args.length > curArg) {
          String UPPER_ARG = args[curArg++].toUpperCase();
    
          if (UPPER_ARG.equals("CUSTOM")) {
            int numClasses = Integer.parseInt(args[curArg++]);
    
            jobClassDescs = new JobClassDescription[numClasses];
            for (int i = 0; i < numClasses; i++) {
              int minW = Integer.parseInt(args[curArg++]);
              int maxW = Integer.parseInt(args[curArg++]);
              int minL = Integer.parseInt(args[curArg++]);
              int maxL = Integer.parseInt(args[curArg++]);
    
              jobClassDescs[i] = new JobClassDescription(minW, maxW, minL, maxL);
            }
    
            fracsOfClasses = new double[numClasses];
            for (int i = 0; i < numClasses; i++) {
              fracsOfClasses[i] = Integer.parseInt(args[curArg++]);
            }
    
            numRacks = Integer.parseInt(args[curArg++]);
            numJobs = Integer.parseInt(args[curArg++]);
            randomSeed = Integer.parseInt(args[curArg++]);
    
            traceProducer = new CustomTraceProducer(numRacks, numJobs, jobClassDescs, fracsOfClasses,
                randomSeed);
          } else if (UPPER_ARG.equals("COFLOW-BENCHMARK")) {
            String pathToCoflowBenchmarkTraceFile = args[curArg++];
            traceProducer = new CoflowBenchmarkTraceProducer(pathToCoflowBenchmarkTraceFile);
          }
        }
        traceProducer.prepareTrace();
    
        // sharingAlgo = SHARING_ALGO.DARK;
        Simulator nlpl = null;
        if (sharingAlgo == SHARING_ALGO.FAIR || sharingAlgo == SHARING_ALGO.PFP) {
          nlpl = new FlowSimulator(sharingAlgo, traceProducer, isOffline, considerDeadline,
              deadlineMultRandomFactor);
        } else if (sharingAlgo == SHARING_ALGO.DARK) {
          nlpl = new CoflowSimulatorDark(sharingAlgo, traceProducer);
        } else {
          nlpl = new CoflowSimulator(sharingAlgo, traceProducer, isOffline, considerDeadline,
              deadlineMultRandomFactor);
        }
        this.simulator = nlpl;
        // System.out.println("Simulator Class: "+nlpl.getClass().getName());
    }

    public static void main(String[] args) {
        CoflowGym gym = new CoflowGym(args);
        double initVal = 10485760.0;
        double[] thresholds = new double[9];
        thresholds[0] = initVal;
        for (int i = 1; i < 9; ++i) {
            thresholds[i] = thresholds[i-1]*10;
        }
        System.out.println(Arrays.toString(thresholds));
        for (int k = 0; k < 2; ++k) {
            gym.reset();
            for (int i = 0;i < 400; ++i) {
                String res = gym.toOneStep(thresholds);
                // System.out.println("Step: "+res);
                gym.takeAction(thresholds);
                // if (res) break;
            }
            System.out.println("\nResult: ");
            gym.printStats();
        }
    }
}