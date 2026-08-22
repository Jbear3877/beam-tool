"""
BeamSolver - 2D梁有限元求解器
支持: 固支/铰支/竖向约束 边界条件
载荷: 集中力(Fx, Fy, Mz) + 分布载荷(线性变化)
"""

import numpy as np


class BeamSolver:
    def __init__(self):
        self.nodes = []          # 节点x坐标 [mm]
        self.elements = []       # [(ni, nj, Iy, A, E), ...]
        self.supports = {}       # node_id -> type
        self.point_loads = {}    # node_id -> (Fx, Fy, Mz)
        self.dist_loads = []     # [(elem_id, w_start, w_end), ...]
        self.reactions = {}      # node_id -> {Fx, Fy, Mz}
        self._U = None

    def add_node(self, x):
        node_id = len(self.nodes)
        self.nodes.append(x)
        return node_id

    def add_element(self, node_i, node_j, Iy, A, E):
        elem_id = len(self.elements)
        self.elements.append((node_i, node_j, Iy, A, E))
        return elem_id

    def add_support(self, node_id, support_type):
        self.supports[int(node_id)] = support_type

    def add_point_load(self, node_id, Fx, Fy, Mz):
        self.point_loads[int(node_id)] = (Fx, Fy, Mz)

    def add_dist_load(self, elem_id, w_start, w_end):
        self.dist_loads.append((int(elem_id), w_start, w_end))

    def solve(self):
        n_nodes = len(self.nodes)
        n_dof = n_nodes * 3
        K = np.zeros((n_dof, n_dof))
        F = np.zeros(n_dof)

        for (ni, nj, Iy, A, E) in self.elements:
            L = self.nodes[nj] - self.nodes[ni]
            EI = E * Iy
            ke = np.zeros((6, 6))

            ke[0, 0] =  E * A / L
            ke[0, 3] = -E * A / L
            ke[3, 0] = -E * A / L
            ke[3, 3] =  E * A / L

            ke[1, 1] =  12 * EI / L**3
            ke[1, 2] =   6 * EI / L**2
            ke[1, 4] = -12 * EI / L**3
            ke[1, 5] =   6 * EI / L**2
            ke[2, 1] =   6 * EI / L**2
            ke[2, 2] =   4 * EI / L
            ke[2, 4] =  -6 * EI / L**2
            ke[2, 5] =   2 * EI / L
            ke[4, 1] = -12 * EI / L**3
            ke[4, 2] =  -6 * EI / L**2
            ke[4, 4] =  12 * EI / L**3
            ke[4, 5] =  -6 * EI / L**2
            ke[5, 1] =   6 * EI / L**2
            ke[5, 2] =   2 * EI / L
            ke[5, 4] =  -6 * EI / L**2
            ke[5, 5] =   4 * EI / L

            dofs = [3*ni, 3*ni+1, 3*ni+2, 3*nj, 3*nj+1, 3*nj+2]
            for r in range(6):
                for c in range(6):
                    K[dofs[r], dofs[c]] += ke[r, c]

        for nid, (Fx, Fy, Mz) in self.point_loads.items():
            F[3*nid]     += Fx
            F[3*nid + 1] += Fy
            F[3*nid + 2] += Mz

        for (eid, w1, w2) in self.dist_loads:
            ni, nj, _, _, _ = self.elements[eid]
            L = self.nodes[nj] - self.nodes[ni]
            F_i = (7*w1 + 3*w2) * L / 20.0
            M_i = (3*w1 + 2*w2) * L**2 / 60.0
            F_j = (3*w1 + 7*w2) * L / 20.0
            M_j = -(2*w1 + 3*w2) * L**2 / 60.0
            F[3*ni + 1] += F_i
            F[3*ni + 2] += M_i
            F[3*nj + 1] += F_j
            F[3*nj + 2] += M_j

        fixed_dofs = set()
        for nid, stype in self.supports.items():
            if stype == 'fixed':
                fixed_dofs.update([3*nid, 3*nid+1, 3*nid+2])
            elif stype == 'pinned':
                fixed_dofs.update([3*nid, 3*nid+1])
            elif stype == 'roller_y':
                fixed_dofs.add(3*nid + 1)

        fixed_dofs = sorted(fixed_dofs)
        free_dofs = sorted(set(range(n_dof)) - fixed_dofs)

        if len(free_dofs) == 0:
            raise ValueError("没有自由度")

        Kff = K[np.ix_(free_dofs, free_dofs)]
        Kfc = K[np.ix_(free_dofs, fixed_dofs)]
        Ff = F[free_dofs]

        U = np.zeros(n_dof)
        U[free_dofs] = np.linalg.solve(Kff, Ff - Kfc @ U[fixed_dofs])

        R = K @ U - F
        self.reactions = {}
        for nid in self.supports:
            self.reactions[nid] = {
                'Fx': float(R[3*nid]),
                'Fy': float(R[3*nid + 1]),
                'Mz': float(R[3*nid + 2]),
            }

        self._U = U
        return U

    def get_internal_forces(self, n_points=100):
        U = self._U
        if U is None:
            raise ValueError("请先调用 solve()")

        results = []

        for eid, (ni, nj, Iy, A, E) in enumerate(self.elements):
            L = self.nodes[nj] - self.nodes[ni]
            EI = E * Iy
            x_local = np.linspace(0, L, n_points)
            x_global = x_local + self.nodes[ni]

            uy_i = U[3*ni + 1]
            rz_i = U[3*ni + 2]
            uy_j = U[3*nj + 1]
            rz_j = U[3*nj + 2]

            w1, w2 = 0.0, 0.0
            for (eid_d, ws, we) in self.dist_loads:
                if eid_d == eid:
                    w1, w2 = ws, we

            if abs(w1) < 1e-15 and abs(w2) < 1e-15:
                moment = np.zeros(n_points)
                shear = np.zeros(n_points)
                deflection = np.zeros(n_points)

                for j, x in enumerate(x_local):
                    xi = x / L
                    N1 = 1 - 3*xi**2 + 2*xi**3
                    N2 = L * (xi - 2*xi**2 + xi**3)
                    N3 = 3*xi**2 - 2*xi**3
                    N4 = L * (-xi**2 + xi**3)

                    N1_dd = (-6 + 12*xi) / L**2
                    N2_dd = (-4 + 6*xi) / L
                    N3_dd = (6 - 12*xi) / L**2
                    N4_dd = (-2 + 6*xi) / L

                    N1_ddd =  12 / L**3
                    N2_ddd =   6 / L**2
                    N3_ddd = -12 / L**3
                    N4_ddd =   6 / L**2

                    v = N1*uy_i + N2*rz_i + N3*uy_j + N4*rz_j
                    v_dd  = N1_dd*uy_i + N2_dd*rz_i + N3_dd*uy_j + N4_dd*rz_j
                    v_ddd = N1_ddd*uy_i + N2_ddd*rz_i + N3_ddd*uy_j + N4_ddd*rz_j

                    deflection[j] = v
                    moment[j] = EI * v_dd
                    shear[j] = EI * v_ddd
            else:
                vp_L = (w1*L**4)/(24*EI) + (w2-w1)*L**4/(120*EI)
                vp_prime_L = (w1*L**3)/(6*EI) + (w2-w1)*L**3/(24*EI)

                A_mat = np.array([[L**2, L**3], [2*L, 3*L**2]])
                b_vec = np.array([uy_j - uy_i - rz_i*L - vp_L,
                                  rz_j - rz_i - vp_prime_L])
                C3, C4 = np.linalg.solve(A_mat, b_vec)
                C1, C2 = uy_i, rz_i

                deflection = np.zeros(n_points)
                moment = np.zeros(n_points)
                shear = np.zeros(n_points)

                for j, x in enumerate(x_local):
                    vp = (w1*x**4)/(24*EI) + (w2-w1)*x**5/(120*EI*L)
                    vp_dd = (w1*x**2)/(2*EI) + (w2-w1)*x**3/(6*EI*L)
                    vp_ddd = (w1*x)/(EI) + (w2-w1)*x**2/(2*EI*L)

                    v = C1 + C2*x + C3*x**2 + C4*x**3 + vp
                    v_dd = 2*C3 + 6*C4*x + vp_dd
                    v_ddd = 6*C4 + vp_ddd

                    deflection[j] = v
                    moment[j] = EI * v_dd
                    shear[j] = EI * v_ddd

            results.append({
                'elem_id': eid,
                'x_global': x_global,
                'moment': moment * 1e-6,
                'shear': shear * 1e-3,
                'deflection': deflection,
            })

        return results

    def get_max_results(self, internal_forces):
        max_moment = max_shear = max_deflection = 0.0
        for r in internal_forces:
            max_moment = max(max_moment, np.max(np.abs(r['moment'])))
            max_shear = max(max_shear, np.max(np.abs(r['shear'])))
            max_deflection = max(max_deflection, np.max(np.abs(r['deflection'])))
        return {
            'max_moment': max_moment,
            'max_shear': max_shear,
            'max_deflection': max_deflection,
        }
