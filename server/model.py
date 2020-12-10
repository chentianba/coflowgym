from coflowgym.util import prepare_pm, KDE
from coflowgym.algo.ddpg import DDPG
from coflowgym.train import action_with_kde

class DDPGModel:
    def __init__(self, a_dim, s_dim, a_bound, model):
        self.agent = DDPG(a_dim, s_dim, a_bound)
        # self.agent.load("%s/model_%s.ckpt"%(mdir, model))
        self.agent.load(model)
        self.kde = KDE(prepare_pm())

    def getAction(self, obs):
        action = self.agent.choose_action(obs)
        return action_with_kde(self.kde, action)
