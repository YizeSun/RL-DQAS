from copy import deepcopy
import pennylane as qml
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from itertools import zip_longest
from enum import Enum


class Learning_State(Enum):
    LEARNING = 1
    FINISH = 0

class Circuit_manager():
    def __init__(self, sphc_struc, sphc_ranges, num_qubits, num_placeholders, num_layers, ops, noisy=False, learning_step=2, topK=2, learning_state=Learning_State.LEARNING.value):
        """
        current_layer_struc is changed by generator(tmp change) and updating(perpetual)
        learned_layer_struc contains top 2 gats for each leanring placeholder e.g. ["H,RZ", "RY", "E", "RZ", "E"]
        """
        self.learned_struc = []
        self.learned_ranges = []

        # sphc
        self._sphc_struc = sphc_struc
        self.sphc_ranges = sphc_ranges
        # learning state
        self.learning_state = learning_state # must before init_struc
        # currents
        self.init_struc(num_placeholders=num_placeholders
                        , sphc_struc=sphc_struc
                        , sphc_ranges=sphc_ranges
                        , num_qubits=num_qubits
                        , layer_struc_parser=self._init_layer_struc_parser)
        # fixed numbers
        self._NUM_PLACEHOLDERS = len(self.learning_places)
        self._NUM_BLOCKS = len(self.current_layer_struc)
        self._NUM_QUBITS = num_qubits
        self._NUM_LAYERS = num_layers
        self._OPS = ops
        self.GATE_KEY = {}
        for k, v in ops.items():
            self.GATE_KEY[v[0]] = k
        self._NUM_OPS = len(ops)
        self._NUM_OPS_W = len(ops) + 1 if len(sphc_struc) else len(ops) # op1, op2, op3 | op4 (from sphc_struc)
        self._TOPK = topK
        self._LEARNING_STEP = learning_step
        self._NOISY = noisy

        # currents
        self.indexs = [self.num_ops_w - 1 for _ in range(self.num_blocks)]
        self.current_learning_places = self.learning_places[:self._LEARNING_STEP] #  order is important
        self.remaining_learning_places = self.learning_places[self._LEARNING_STEP:]
        self._current_num_placeholders = len(self.current_learning_places)

        # learned struc
        self.learned_layer_struc = deepcopy(self.current_layer_struc)
        self.learned_layer_ranges = deepcopy(self.current_layer_ranges)
        print(f"current learning layer struc: {self.current_layer_struc}")
        print(f"current learning layer ranges: {self.current_layer_ranges}")
        print(f"current learning places: {self.current_learning_places}")
        print(f"remaining learning places: {self.remaining_learning_places}")
        print(f"current num placeholders: {self.current_num_placeholders}")
        # current learning layer struc: ['E', 'RY', 'E', 'RZ', 'E', 'CNOT']
        # current learning layer ranges: [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
        # current learning places: [0, 2]
        # remaining learning places: [4]
        # current num placeholders: 2

        # var gates
        self.var_gates = ["U1", "U3", "CU3", "RX", "RY", "RZ", "CRZ", "ZZ"]
        # sampled struc
        self.current_sampled_struc = None
    
    def set_current_sampled_struc(self, value):
        # current_sampled_struc: [1 3]
        self.current_sampled_struc = value
    
    def get_current_sampled_struc(self):
        # current_sampled_struc: [1 3]
        return self.current_sampled_struc

    def add_learned_layer_struc(self, struc, ranges, places):
        self.learned_struc.append((struc.copy(), ranges.copy(), places.copy()))

    def init_struc(self, num_placeholders, sphc_struc, sphc_ranges, num_qubits, layer_struc_parser):
        """
        The learning circuit is described by 
            1. current_layer_struc: ["E", "RZ", "E", "CNOT", "E"] end with "E" iff num_placeholders > len(sphc_struc) 
            2. current_layer_ranges: [[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3]]
            3. learning_places [0, 2]
        """
        self.current_layer_struc, self.current_layer_ranges, self.learning_places = layer_struc_parser(num_placeholders=num_placeholders
                                                                                                        , sphc_struc=sphc_struc
                                                                                                        , sphc_ranges=sphc_ranges
                                                                                                        , num_qubits=num_qubits)

    def update_learning_places(self, prob):
        self._update_struc(prob)
        self.current_learning_places = self.remaining_learning_places[:self._LEARNING_STEP]
        self.remaining_learning_places = self.remaining_learning_places[self._LEARNING_STEP:]
        self._current_num_placeholders = len(self.current_learning_places)

        print(f"current learning layer struc: {self.current_layer_struc}")
        print(f"current learning layer ranges: {self.current_layer_ranges}")
        print(f"current learning places: {self.current_learning_places}")
        print(f"remaining learning places: {self.remaining_learning_places}")
        print(f"current num placeholders: {self.current_num_placeholders}")
        print(f"learned_layer_struc: {self.learned_layer_struc}") # learned_layer_struc: ['CNOT,RY', 'RY', 'RZ,CNOT', 'RZ', 'CNOT,RY', 'CNOT']

        if self.current_num_placeholders > 0:
            return False
        else:
            # current learning layer struc: ['CNOT', 'CNOT', 'CNOT']
            # current learning layer ranges: [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
            # current learning places: []
            # remaining learning places: []
            # current num placeholders: 0
            # learned_layer_struc: ['CNOT,RZ', 'CNOT,RZ', 'CNOT,RZ']
            self.learning_state = Learning_State.FINISH.value
            return True

    def _update_struc(self, prob):

        for i, current_plh_i in enumerate(prob):
            _, best_gate_idxs = torch.topk(current_plh_i, k=self._TOPK)
            if i%len(prob)==0:
                print(f"best gate idx: {best_gate_idxs}")
            self.current_layer_struc[self.current_learning_places[i]] = self.ops[best_gate_idxs[0].item()][0]
            self.current_layer_ranges[self.current_learning_places[i]] = self.ops[best_gate_idxs[0].item()][1]
            self.learned_layer_struc[self.current_learning_places[i]] = ",".join(self.ops[j.item()][0] for j in best_gate_idxs)
            self.learned_layer_ranges[self.current_learning_places[i]] = [self.ops[j.item()][1] for j in best_gate_idxs]
            self.indexs[self.current_learning_places[i]] = best_gate_idxs[0].item()
        print(f"learned layer struc: {self.learned_layer_struc} learned layer ranges: {self.learned_layer_ranges}, indexs: {self.indexs}")

    def _init_layer_struc_parser(self, num_placeholders, sphc_struc, sphc_ranges, num_qubits):
        if num_placeholders > 0:
            tmp_layer_struc = zip_longest(["E" for _ in range(num_placeholders)], sphc_struc, fillvalue="")
            layer_struc = [gate for t in tmp_layer_struc for gate in t if len(gate) > 0] # [E, E, E]
            layer_ranges = [[*range(num_qubits)] for _ in range(len(layer_struc))]
            learning_places = [i for i, plh in enumerate(layer_struc) if plh == "E"]
        else:
            layer_struc = deepcopy(sphc_struc)
            layer_ranges = deepcopy(sphc_ranges)
            learning_places = []
            self.learning_state = Learning_State.FINISH.value
        # e.g. ["E", "RY", "E", "RZ", "E", "CNOT", "E"] if n_plds > n_sphc

        return layer_struc, layer_ranges, learning_places

    def get_layer_generator(self, gates, ranges=[]):
        # gns = [1,0,1]
        def make_layer_generator(gates, num_blocks, ops, num_ops_w, current_layer_struc, current_layer_ranges, current_learning_places, ranges, indexs):
            for i, gate in enumerate(gates):
                current_layer_struc[current_learning_places[i]] = ops[gate][0]
                current_layer_ranges[current_learning_places[i]] = ops[gate][1]
                if len(ranges):
                    assert(len(ranges)==len(gates))
                    assert(len(ranges[i])>0)
                    current_layer_ranges[current_learning_places[i]] = ranges[i]
                indexs[current_learning_places[i]] = gate

            # print(f"current_layer_struc: {current_layer_struc}, current_layer_ranges: {current_layer_ranges}, indexs: {indexs}")
            return current_layer_struc, current_layer_ranges, indexs # ["U3", "RY", "CU3", "RZ", "U3", "CNOT", "U3"]
        
        def make_learned_layer_generator(num_blocks, num_ops_w, learning_places, learned_layer_struc, learned_layer_ranges, gate_key, indexs):
            # TODO: get best struc from self.get_learned_layer and make some indexs
            # take only first operator
            # indexs = [num_ops_w - 1 for _ in range(num_blocks)]
            # for i in learning_places:
            #     indexs[i] = gate_key[learned_layer_struc[i]]

            return learned_layer_struc, learned_layer_ranges, indexs

        if self.learning_state == Learning_State.LEARNING.value:
            return make_layer_generator(gates=gates
                                    , num_blocks=self.num_blocks
                                    , ops=self.ops
                                    , num_ops_w=self.num_ops_w
                                    , current_layer_struc=self.current_layer_struc
                                    , current_layer_ranges=self.current_layer_ranges
                                    , current_learning_places=self.current_learning_places
                                    , ranges=ranges
                                    , indexs=self.indexs
                                    )
        else:
            return make_learned_layer_generator(num_blocks=self.num_blocks
                                            , num_ops_w=self.num_ops_w
                                            , learning_places=self.learning_places
                                            , learned_layer_struc=self.current_layer_struc
                                            , learned_layer_ranges=self.current_layer_ranges
                                            , gate_key=self.GATE_KEY
                                            , indexs=self.indexs
                                            )

    def get_weights_shape(self):
        return (self.num_layers, self.num_qubits, self.num_blocks, self.num_ops_w)
    
    def check_gate(self, gate:str):
        return gate in self.var_gates
        # if len(ops): # check gate(int) with operation pool
        #     return ops[gate][0] in self.var_gates
        # else: # check gate(str) with sphc pool
        #     return gate in self.var_gates
    
    def get_learned_layer_struc(self):
        return self.learned_layer_struc, self.learned_layer_ranges

    @property
    def num_ops(self):
        return self._NUM_OPS

    @property
    def num_ops_w(self):
        return self._NUM_OPS_W

    @property
    def num_blocks(self): # caution ! check this for every new current layer struc
        return self._NUM_BLOCKS

    @property
    def num_qubits(self):
        return self._NUM_QUBITS

    @property
    def num_layers(self):
        return self._NUM_LAYERS

    @property
    def noisy(self):
        return self._NOISY

    @property
    def sphc_struc(self):
        return self._sphc_struc

    # @sphc_struc.setter
    # def sphc_struc(self, value):
    #     self._sphc_struc = value

    @property
    def current_num_placeholders(self):
        return self._current_num_placeholders
    
    @property
    def num_placeholders(self):
        return self._NUM_PLACEHOLDERS

    @property
    def ops(self):
        return self._OPS

    # @ops.setter
    # def ops(self, value):
    #     self._OPS = value

    @property
    def learning_step(self):
        return self._LEARNING_STEP

    # @learning_step.setter
    # def learning_step(self, value):
    #     self._LEARNING_STEP = value

def circuit(cm:Circuit_manager, data_reuploading=False, barrier=False): # keep it outside DQN
        #dev = qml.device('qiskit.aer', wires=num_qubits)
        if cm.noisy:
            dev = qml.device("default.mixed", wires=cm.num_qubits)
        else:
            dev = qml.device("default.qubit", wires=cm.num_qubits)
        #dev = qml.device("cirq.simulator", wires=num_qubits)

        # U3 weights
        shapes = {
            "theta_weights": cm.get_weights_shape(),
            "phi_weights": cm.get_weights_shape(),
            "delta_weights": cm.get_weights_shape(),
        }

        def encoding_block(inputs, num_qubits): # for cart pole
            for w in range(num_qubits):
                qml.RX(inputs[w], wires=w)
            # for w in range(num_qubits):
            #     qml.RZ(inputs[w], wires=w)

        def measure_block(num_qs):# TODO: for two qubits frozen lake
            return [qml.expval(qml.PauliZ(k)) for k in range(num_qs)]

        def entanglement(rg):
            for i in rg:
                qml.CNOT(wires=[i, (i + 1 ) % len(rg)])

        def layer(theta_weight, phi_weight, delta_weight, generators, generators_range, indexs):
            # p: 0, op: U3, pr: [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], gnr: ['U3', 'RY', 'CU3', 'RZ', 'U3', 'CNOT'], op range: [0, 1, 2, 3]
            for p, op in enumerate(generators):
                # if p%len(generators)==0:
                #     print(f"p: {p}, op: {op}, pr: {generators_range}, gnr: {generators}, op range: {generators_range[p]}, index p :{indexs[p]}") #
                op_range = generators_range[p]
                # -- u3 & cu3 gate --
                if op == "U3":
                    for i in op_range:
                        qml.U3(theta=theta_weight[i, p, indexs[p]] # indexs: [0, 1, 0, 2, 2, 2]
                            , phi=phi_weight[i, p, indexs[p]]
                            , delta=delta_weight[i, p, indexs[p]]
                            , wires=i)
                elif op == "CU3":
                    assert(len(op_range) > 1)
                    for i in op_range:
                        # if i + 1 in op_range:
                            def CU3(t_w, p_w, d_w, target):
                                qml.U3(theta=t_w[i, p, indexs[p]]
                                    , phi=p_w[i, p, indexs[p]]
                                    , delta=d_w[i, p, indexs[p]]
                                    , wires=target)
                            qml.ctrl(CU3, control=i)(theta_weight, phi_weight, delta_weight, (i+1)%len(op_range))
                # -- simple one qubit gate --
                elif op == "U1":
                    for i in op_range:
                        qml.U1(theta_weight[i, p, indexs[p]], wires=i)
                elif op == "RX":
                    for i in op_range:
                        qml.RX(theta_weight[i, p, indexs[p]], wires=i)
                elif op == "RY":
                    for i in op_range:
                        qml.RY(theta_weight[i, p, indexs[p]], wires=i)
                elif op == "RZ":
                    # print(f"RZ: {p}, {indexs[p]}")
                    for i in op_range:
                        qml.RZ(theta_weight[i, p, indexs[p]], wires=i)
                elif op == "X":
                    for i in op_range:
                        qml.X(wires=i)
                elif op == "SX":
                    for i in op_range:
                        qml.SX(wires=i)
                elif op == "T":
                    for i in op_range:
                        qml.T(wires=i)
                elif op  == "H":
                    for i in op_range:
                        qml.Hadamard(wires=i)
                elif op == "SQH":
                    unitary = (1./4+1.j/4) * np.array([-1.j*(np.sqrt(2)+2.j), -1.j*np.sqrt(2)],
                                                    [-1.j*np.sqrt(2), 1.j*(np.sqrt(2)-2.j)])
                    for i in op_range:
                        qml.QubitUnitary(unitary, wires=[i])
                elif op == "S":
                    for i in op_range:
                        qml.S(wires=i)
                elif op == "E":
                    for i in op_range:
                        qml.Identity(wires=i)
                # -- two qubits gate --
                elif op == "SWAP":
                    qml.SWAP(wires=op_range)
                elif op == "SQSWAP":
                    unitary = np.array([1., 0., 0., 0.],
                                        [0., (1./2+1.j/2), (1./2-1.j/2), 0.],
                                        [0., (1./2-1.j/2), (1./2+1.j/2), 0.],
                                        [0., 0., 0., 1.],)
                    qml.QubitUnitary(unitary, wires=op_range)
                elif op == "CZ":
                    assert(len(op_range) > 1)
                    for i in op_range:
                        if i + 1 in op_range:
                            qml.CZ(wires=[i, (i + 1) % len(op_range)])
                elif op == "CRZ":
                    assert(len(op_range) > 1)
                    for i in op_range:
                        if i + 1 in op_range:
                            qml.CRZ(theta_weight[(i + 1) % len(op_range), p, indexs[p]], wires=[i, (i + 1) % len(op_range)])
                elif op == "CNOT":
                    assert(len(op_range) > 1)
                    entanglement(op_range)
                elif op == "ZZ":
                    assert(len(op_range) > 1)
                    for i in op_range:
                        if i + 1 in op_range:
                            qml.IsingZZ(theta_weight[(i + 1) % len(op_range), p, indexs[p]], wires=[i, (i + 1) % len(op_range)])
                elif op == "QFT":
                    assert(len(op_range) > 1)
                    qml.QFT(wires=op_range)
        
        # !!! The args name "inputs", "generators" should not be edited.
        # the source code of pennylane was modified for passing structure learning: see pennylane/qnn/torch.py
        def make_circuit(inputs, theta_weights, phi_weights, delta_weights):
            # encoding_block(inputs[:4], num_qubits)
            # qml.Barrier(wires=range(num_qubits))
            # inputs : tensor([-0.0187, -0.0415,  0.0539,  0.0258,  0.0000,  0.0000,  2.0000], dtype=torch.float64)
            # print(f"inputs: {inputs[4:]}")
            
            # generators / placehoders
            # if len(sphc_struc):
            #     tmp_gns = zip_longest(map(__get_operation, map(int, inputs[4:])), sphc_struc, fillvalue="Id")
            #     gns = [g for t in tmp_gns for g in t]
            #     tmp_plh_rs = zip_longest(placeholder_range, [[*range(num_qubits)] for _ in range(len(sphc_struc))], fillvalue=[*range(num_qubits)])
            #     plh_rs = [plh_r for t in tmp_plh_rs for plh_r in t]
            #     tmp_indexs = zip_longest(map(int, inputs[4:]), [num_ops for _ in range(len(sphc_struc))], fillvalue=num_ops)
            #     indexs = [idx for t in tmp_indexs for idx in t]
            # else:
            #     gns = [*map(__get_operation, map(int, inputs[4:]))]
            #     plh_rs = placeholder_range
            #     indexs = [*map(int, inputs[4:])]

            gns, gns_ranges, indexs = cm.get_layer_generator(gates=cm.get_current_sampled_struc())

            # gns: ['RZ', 'CNOT', 'CNOT']
            # gns rs: [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
            # indexs: [0, 1, 1]
            # print(f"gns: {gns}") # gns: ['RY', 'RY', 'RZ', 'RZ', 'CNOT', 'CNOT'] | gns: ['RZ', 'CNOT', 'CNOT'] for sphc_struc = []
            # print(f"gns rs: {gns_ranges}") # plh rs: [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]] 
            # | gns rs: [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
            # print(f"indexs: {indexs}") # indexs: [0, 1, 1, 3, 3, 3] | [0, 1, 1]
            
            # make circuit layerwise
            for l in range(cm.num_layers):
                if data_reuploading or l==0:
                    encoding_block(inputs, cm.num_qubits)
                if barrier:
                    qml.Barrier(wires=range(cm.num_qubits))
                layer(theta_weights[l], phi_weights[l], delta_weights[l], gns, gns_ranges, indexs)
                if barrier:
                    qml.Barrier(wires=range(cm.num_qubits))
                if cm.noisy:
                    for i in range(cm.num_qubits):
                        qml.BitFlip(0.01, wires=i)
            return measure_block(cm.num_qubits)

        circuit = qml.QNode(make_circuit, dev, interface='torch')
        
        model = qml.qnn.TorchLayer(circuit, shapes)
        
        return model


class QDQN(nn.Module):
    def __init__(self
                , cm:Circuit_manager
                , w_input=False
                , w_output=False
                , data_reuploading=False
                , barrier=False
                , seed=1234):
        super(QDQN, self).__init__()
        torch.manual_seed(seed)
        self.num_qubits = cm.num_qubits
        self.num_actions = 4 #TODO: fit for cart pole
        self.cm = cm
        self.q_layers = circuit(cm=self.cm, data_reuploading=data_reuploading, barrier=barrier)

        if w_input:
            self.w_input = Parameter(torch.Tensor(self.num_qubits))
            nn.init.normal_(self.w_input)
        else:
            self.register_parameter('w_input', None)
        if w_output:
            self.w_output = Parameter(torch.Tensor(self.num_actions))
            nn.init.normal_(self.w_output, mean=90.)
        else:
            self.register_parameter('w_output', None)

    def forward(self, inputs):
        # print(f"** -- circuit forward -- **")
        inputs = inputs*np.pi
        out = self.q_layers(inputs)
        out = (1 + out) / 2

        return out
    
    def set_circuit_struc(self, gates):
        self.cm.set_current_sampled_struc(gates)