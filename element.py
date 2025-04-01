import numpy as np


class Quad:
    
    def __init__(self,n,i,j,k,l,i_node,j_node,k_node,l_node, t, E, poisson, mode=0):
        self.n=n
        self.i=i
        self.j=j
        self.k=k
        self.l=l
        self.i_node=np.array(i_node)
        self.j_node=np.array(j_node)
        self.k_node=np.array(k_node)
        self.l_node=np.array(l_node)
        self.t=t
        self.E=E
        self.poisson=poisson
        
    
        # Variables to be calculated
        self.h_i=0
        self.h_j=0
        self.h_k=0
        self.h_l=0
        self.v_i=0
        self.h_j=0
        self.v_k=0
        self.v_l=0

        # Partial Derivatives of Shape Functions
        N_1s = lambda s, t: 1 / 4 * (t - 1)
        N_1t = lambda s, t: 1 / 4 * (s - 1)
        N_2s = lambda s, t: 1 / 4 * (1 - t)
        N_2t = lambda s, t: -1 / 4 * (1 + s)
        N_3s = lambda s, t: 1 / 4 * (1 + t)
        N_3t = lambda s, t: 1 / 4 * (1 + s)
        N_4s = lambda s, t: -1 / 4 * (1 + t)
        N_4t = lambda s, t: 1 / 4 * (1 - s)

        N_ii = np.array([[N_1s, N_1t],
                         [N_2s, N_2t],
                         [N_3s, N_3t],
                         [N_4s, N_4t]])

        self.N_ii = N_ii
        
        # Definition of constitutive matrix
        # Plain Strain deformation case
        if mode == 1:
            factor = E / ((1 + poisson)*(1 - 2*poisson))
            D = factor * np.array([[1-poisson, poisson, 0],
                                   [poisson, 1-poisson, 0],
                                   [0, 0, (1-2*poisson)/2]],
                                  dtype=float)
        # Case of plane stress
        else:
            factor = E / (1 - poisson**2)
            D = factor * np.array([[1, poisson, 0],
                                   [poisson, 1, 0],
                                   [0, 0, (1 - poisson)/2]],
                                  dtype=float)
        self.D = D

        # Numerical integration of the stiffness matrix
        # Gaussian quadrature for 2 points per direction
        K = np.zeros((8, 8))

        s_list = [-.5773, -.5773, .5773, .5773]
        t_list = [-.5773, .5773, -.5773, .5773]
        W_list = [1, 1, 1, 1]

        for i in range(4):
            s = s_list[i]
            t = t_list[i]
            W = W_list[i]
            jacobian = self.jacobian(s, t)
            B = self.B(s, t)
            D = self.D
            K_i = np.dot(np.dot(B.transpose(),D),B) * self.t * jacobian * W * W
            K += K_i

        self.K = K

    # Factors a, b, c, and d for the gradient matrix "B"
    def a(self, s, t):
        a  = 1 / 4 * (self.i_node[1] * (s - 1) + self.j_node[1] * (-1 - s))
        a += 1 / 4 * (self.k_node[1] * (1 + s) + self.l_node[1] * (1 - s))
        return a

    def b(self, s, t):
        b  = 1 / 4 * (self.i_node[1] * (t - 1) + self.j_node[1] * (1 - t))
        b += 1 / 4 * (self.k_node[1] * (1 + t) + self.l_node[1] * (-1 - t))
        return b

    def c(self, s, t):
        c  = 1 / 4 * (self.i_node[0] * (t - 1) + self.j_node[0] * (1 - t))
        c += 1 / 4 * (self.k_node[0] * (1 + t) + self.l_node[0] * (-1 - t))
        return c

    def d(self, s, t):
        d  = 1 / 4 * (self.i_node[0] * (s - 1) + self.j_node[0] * (-1 - s))
        d += 1 / 4 * (self.k_node[0] * (1 + s) + self.l_node[0] * (1 - s))
        return d

    def B_i(self, s, t, i):
        B_i = np.zeros((3, 2))
        B_i[0, 0] = self.a(s, t) * self.N_ii[i,0](s, t) - self.b(s, t) * self.N_ii[i,1](s, t)
        B_i[1, 1] = self.c(s, t) * self.N_ii[i,1](s, t) - self.d(s, t) * self.N_ii[i,0](s, t)
        B_i[2, 0] = self.c(s, t) * self.N_ii[i,1](s, t) - self.d(s, t) * self.N_ii[i,0](s, t)
        B_i[2, 1] = self.a(s, t) * self.N_ii[i,0](s, t) - self.b(s, t) * self.N_ii[i,1](s, t)
        return B_i

    def jacobian(self, s, t):
        X_c = np.array([self.i_node[0], self.j_node[0], self.k_node[0], self.l_node[0]])
        Y_c = np.array([self.i_node[1], self.j_node[1], self.k_node[1], self.l_node[1]])

        J_11 = np.dot(np.array([-1 + t, 1 - t, 1 + t, -1 - t]), X_c)
        J_12 = np.dot(np.array([-1 + t, 1 - t, 1 + t, -1 - t]), Y_c)
        J_21 = np.dot(np.array([-1 + s, -1 - s, 1 + s, 1 - s]), X_c)
        J_22 = np.dot(np.array([-1 + s, -1 - s, 1 + s, 1 - s]), Y_c)

        J = 1 / 4 * np.array([[J_11, J_12],
                              [J_21, J_22]])

        jacobian = np.linalg.det(J)
        return jacobian

    # Gradient Matrix
    def B(self, s, t):
        B = np.hstack((self.B_i(s, t, 0), self.B_i(s, t, 1), self.B_i(s, t, 2), self.B_i(s, t, 3)))
        B = 1 / self.jacobian(s, t) * B
        return B

    # Determine stresses at the midpoint
    def calc_stress(self):
        B_0 = self.B(0,0)
        q = np.array([self.h_i, self.v_i,
                      self.h_j, self.v_j,
                      self.h_k, self.v_k,
                      self.h_l, self.v_l])
        sigma = np.dot(np.dot(self.D, B_0), q)
        self.sigma = sigma
        return sigma