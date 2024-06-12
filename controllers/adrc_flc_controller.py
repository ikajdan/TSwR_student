import numpy as np

from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManiuplatorModel(Tp, m3=0.1, r3=0.5)
        self.Kp = Kp
        self.Kd = Kd

        self.L = np.array(
            [
                [3 * p, 0],
                [0, 3 * p],
                [3 * p**2, 0],
                [0, 3 * p**2],
                [p**3, 0],
                [0, p**3],
            ]
        )

        W = np.zeros((2, 6))
        W[0:2, 0:2] = np.eye(2)

        A = np.zeros((6, 6))
        A[0:2, 2:4] = np.eye(2)
        A[2:4, 4:6] = np.eye(2)
        A[2:4, 2:4] = np.zeros((2, 2))

        B = np.zeros((6, 2))
        B[2:4, :] = np.zeros((2, 2))

        self.eso = ESO(A, B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        ### TODO Implement procedure to set eso.A and eso.B
        self.eso.A = None
        self.eso.B = None

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement centralized ADRFLC
        return NotImplementedError
