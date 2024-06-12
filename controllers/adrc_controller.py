import numpy as np

from models.manipulator_model import ManiuplatorModel

from .adrc_joint_controller import ADRCJointController
from .controller import Controller


class ADRController(Controller):
    def __init__(self, Tp, params):
        self.model = ManiuplatorModel(Tp, m3=0.1, r3=0.5)

        self.joint_controllers = []
        for param in params:
            self.joint_controllers.append(ADRCJointController(*param, Tp))

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        M = self.model.M(x)
        for i, controller in enumerate(self.joint_controllers):
            controller.set_b(1 / M[i, i])

        u = []
        for i, controller in enumerate(self.joint_controllers):
            u.append(
                controller.calculate_control(
                    [x[i], x[i + 2]], q_d[i], q_d_dot[i], q_d_ddot[i]
                )
            )
        u = np.array(u)[:, np.newaxis]
        return u
