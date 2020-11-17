import tensorflow.compat.v1 as tf
import tensorflow.keras.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear


from configs.q3_nature import config

class NatureQN(Linear):
   
    def get_q_values_op(self, state, scope, reuse=False):
        
        num_actions = self.env.action_space.n
        with tf.variable_scope(scope, reuse=reuse) as _:
            conv1 = layers.Conv2D(32, kernel_size=(8, 8), strides=4, activation='relu')(state)
            conv2 = layers.Conv2D(64, kernel_size=(4, 4), strides=2, activation='relu')(conv1)
            conv3 = layers.Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu')(conv2)
            full_inputs = layers.Flatten()(conv3)
            full_layer = layers.Dense(512, activation='relu')(full_inputs)
            out = layers.Dense(num_actions)(full_layer)
        pass
        
        return out


if __name__ == '__main__':
    env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)