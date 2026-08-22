"""
副车架梁模型快速分析工具 - 核心求解器
基于直接刚度法的Euler-Bernoulli梁有限元求解器
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class BeamSolver:
    """
    2D梁结构有限元求解器
    """

    def __init__(self):
        self.nodes = []
        self.elements = []
        self.supports = []
        self.point_loads = []
        self.dist_loads = []
        self.U = None
        self.reactions = None
        self.elem_forces = None

    def add_node(self, x, y=0.0):
        self.nodes.append([x, y])
        return len(self.nodes) - 1

    def add_element(self, node_i, node_j, Iy, A, E):
        self.elements.append([int(node_i), int(node_j), Iy, A, E])
        return len(self.elements) - 1

    def add_support(self, node_id, support_type):
        self.supports.append([int(node_id), support_type])

    def add_point_load(self, node_id, Fx=0.0, Fy=0.0, Mz=0.0):
        self.point_loads.append([int(node_id), Fx, Fy, Mz])

    def add_dist_load(self, elem_id, w_start, w_end):
        self.dist_loads.append([int(elem_id), w_start, w_end])

    def _element_length(self, elem):
        ni, nj = elem[0], elem[1]
        dx = self.nodes[nj][0] - self.nodes[ni][0]
        dy = self.nodes[nj][1] - self.nodes[ni][1]
        return np.sqrt(dx**2 + dy**2)

    def _element_stiffness(self, L, Iy, A, E):
        EA = E * A
        EI = E * Iy
        k = np.array([
            [ EA/L,       0,          0,       -EA/L,       0,          0        ],
            [ 0,     12*EI/L**3,  6*EI/L**2,    0,    -12*EI/L**3,  6*EI/L**2 ],
            [ 0,      6*EI/L**2,  4*EI/L,       0,     -6*EI/L**2,  2*EI/L   ],
            [-EA/L,      0,          0,        EA/L,       0,          0        ],
            [ 0,    -12*EI/L**3, -6*EI/L**2,    0,     12*EI/L**3, -6*EI/L**2 ],
            [ 0,      6*EI/L**2,  2*EI/L,       0,     -6*EI/L**2,  4*EI/L   ]
        ])
        return k

    def _equivalent_nodal_forces(self, L, w1, w2):
        F1 = w1 * L / 2
        M1 = w1 * L**2 / 12
        F2 = w1 * L / 2
        M2 = -w1 * L**2 / 12
        dw = w2 - w1
        F1 += 3 * dw * L / 20
        M1 += dw * L**2 / 30
        F2 += 7 * dw * L / 20
        M2 += -dw * L**2 / 20
        return np.array([F1, M1, F2, M2])

    def solve(self):
        n_nodes = len(self.nodes)
        n_dof = n_nodes * 3

        K_global = np.zeros((n_dof, n_dof))
        self._elem_data = []

        for i, elem in enumerate(self.elements):
            L = self._element_length(elem)
            Iy, A, E = elem[2], elem[3], elem[4]
            k = self._element_stiffness(L, Iy, A, E)
            ni, nj = elem[0], elem[1]
            dofs = [ni*3, ni*3+1, ni*3+2, nj*3, nj*3+1, nj*3+2]
            for r in range(6):
                for c in range(6):
                    K_global[dofs[r], dofs[c]] += k[r, c]
            self._elem_data.append({
                'length': L, 'Iy': Iy, 'A': A, 'E': E,
                'nodes': [ni, nj], 'dofs': dofs, 'k': k
            })

        F_global = np.zeros(n_dof)

        for load in self.point_loads:
            nid, fx, fy, mz = load
            F_global[int(nid)*3]     += fx
            F_global[int(nid)*3 + 1] += fy
            F_global[int(nid)*3 + 2] += mz

        self._elem_dist_forces = [np.zeros(4) for _ in self.elements]

        for dload in self.dist_loads:
            eid, w1, w2 = dload
            L = self._elem_data[int(eid)]['length']
            equiv = self._equivalent_nodal_forces(L, w1, w2)
            dofs = self._elem_data[int(eid)]['dofs']
            F_global[dofs[1]] += equiv[0]
            F_global[dofs[2]] += equiv[1]
            F_global[dofs[4]] += equiv[2]
            F_global[dofs[5]] += equiv[3]
            self._elem_dist_forces[int(eid)] = equiv

        fixed_dofs = []
        for sup in self.supports:
            nid, stype = int(sup[0]), sup[1]
            if stype == 'fixed':
                fixed_dofs.extend([nid*3, nid*3+1, nid*3+2])
            elif stype == 'pinned':
                fixed_dofs.extend([nid*3, nid*3+1])
            elif stype == 'roller_x':
                fixed_dofs.append(nid*3)
            elif stype == 'roller_y':
                fixed_dofs.append(nid*3+1)

        free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]
        free_dofs = np.array(free_dofs)
        fixed_dofs = np.array(sorted(fixed_dofs))

        K_ff = K_global[np.ix_(free_dofs, free_dofs)]
        F_ff = F_global[free_dofs]

        self.U = np.zeros(n_dof)

        if len(free_dofs) > 0:
            K_sparse = csr_matrix(K_ff)
            self.U[free_dofs] = spsolve(K_sparse, F_ff)

        self.reactions = {}
        if len(fixed_dofs) > 0:
            F_reaction = K_global @ self.U - F_global
            for dof in fixed_dofs:
                nid = dof // 3
                comp = dof % 3
                labels = ['Fx', 'Fy', 'Mz']
                if nid not in self.reactions:
                    self.reactions[nid] = {}
                self.reactions[nid][labels[comp]] = F_reaction[dof]

        return self.U

    def get_internal_forces(self, n_points=100):
        results = []

        for eid, elem in enumerate(self.elements):
            ni, nj = elem[0], elem[1]
            L = self._elem_data[eid]['length']
            E = self._elem_data[eid]['E']
            Iy = self._elem_data[eid]['Iy']
            EI = E * Iy

            dofs = self._elem_data[eid]['dofs']
            u_e = self.U[dofs]
            v1, theta1 = u_e[1], u_e[2]
            v2, theta2 = u_e[4], u_e[5]

            eid_doforce = self._elem_dist_forces[eid]
            x_start = self.nodes[ni][0]

            x_local = np.linspace(0, L, n_points)
            x_global = x_local + x_start

            xi = x_local / L

            d2N1 = (-6/L**2 + 12*x_local/L**3)
            d2N2 = (-4/L + 6*x_local/L**2)
            d2N3 = (6/L**2 - 12*x_local/L**3)
            d2N4 = (-2/L + 6*x_local/L**2)

            M_elastic = EI * (d2N1*v1 + d2N2*theta1 + d2N3*v2 + d2N4*theta2)

            equiv = eid_doforce
            w1, w2 = 0, 0
            if abs(equiv[0]) > 1e-10 or abs(equiv[2]) > 1e-10:
                A_mat = np.array([
                    [L/2, 3*L/20],
                    [L/2, 7*L/20]
                ])
                b_vec = np.array([equiv[0], equiv[2]])
                if abs(np.linalg.det(A_mat)) > 1e-15:
                    sol = np.linalg.solve(A_mat, b_vec)
                    w1 = sol[0]
                    w2 = w1 + sol[1]

            V_dist = w1 * x_local + (w2 - w1) * x_local**2 / (2*L)
            M_dist = w1 * x_local**2 / 2 + (w2 - w1) * x_local**3 / (6*L)

            d3N1 = 12/L**3
            d3N2 = 6/L**2
            d3N3 = -12/L**3
            d3N4 = 6/L**2

            V_elastic = EI * (d3N1*v1 + d3N2*theta1 + d3N3*v2 + d3N4*theta2)

            M_total = M_elastic + M_dist
            V_total = V_elastic + V_dist

            results.append({
                'elem_id': eid,
                'x_global': x_global,
                'moment': M_total / 1e3,
                'shear': V_total / 1e3,
                'deflection': np.array([
                    v1*(1-3*xi_t**2+2*xi_t**3) +
                    L*xi_t*(1-xi_t)**2*theta1 +
                    xi_t**2*(3-2*xi_t)*v2 +
                    L*xi_t**2*(xi_t-1)*theta2 for xi_t in xi
                ]),
                'max_moment': np.max(np.abs(M_total)) / 1e3,
                'max_shear': np.max(np.abs(V_total)) / 1e3,
            })

        return results

    def get_node_displacements(self):
        results = []
        for i in range(len(self.nodes)):
            results.append({
                'node': i,
                'x': self.nodes[i][0],
                'ux': self.U[i*3],
                'uy': self.U[i*3+1],
                'rz': self.U[i*3+2],
            })
        return results

    def get_max_results(self, internal_forces):
        max_m = 0
        max_v = 0
        max_d = 0
        for r in internal_forces:
            max_m = max(max_m, r['max_moment'])
            max_v = max(max_v, r['max_shear'])
            max_d = max(max_d, np.max(np.abs(r['deflection'])))
        return {
            'max_moment': max_m,
            'max_shear': max_v,
            'max_deflection': max_d,
        }
