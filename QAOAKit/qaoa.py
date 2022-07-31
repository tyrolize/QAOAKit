# QAOA circuits

import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.compiler import transpile
import numpy as np
import math


def append_zz_term(qc, q1, q2, gamma):
    qc.cx(q1, q2)
    qc.rz(2 * gamma, q2)
    qc.cx(q1, q2)


def append_x_term(qc, q1, beta):
    qc.h(q1)
    qc.rz(2 * beta, q1)
    qc.h(q1)


def append_zzzz_term(qc, q1, q2, q3, q4, gamma):
    qc.cx(q1,q2)
    qc.cx(q2,q3)
    qc.cx(q3,q4)
    qc.rz(2 * gamma, q4)
    qc.cx(q3,q4)
    qc.cx(q2,q3)
    qc.cx(q1,q2)


def append_4_pauli_exponential_term(qc, q1, q2, q3, q4, gamma, pauli="zzzz"):
    allowed_symbols = set("xyz")
    if set(pauli).issubset(allowed_symbols) and len(pauli) == 4:
        if s[0] == "x":
            qc.h(q1)
        elif s[0] == "y":
            qc.y(q1)
        if s[1] == "x":
            qc.h(q2)
        elif s[1] == "y":
            qc.y(q2)
        if s[2] == "x":
            qc.h(q3)
        elif s[2] == "y":
            qc.y(q3)
        if s[3] == "x":
            qc.h(q4)
        elif s[3] == "y":
            qc.y(q4)
        append_zzzz_term(qc, q1, q2, q3, q4, gamma)
        if s[0] == "x":
            qc.h(q1)
        elif s[0] == "y":
            qc.y(q1)
        if s[1] == "x":
            qc.h(q2)
        elif s[1] == "y":
            qc.y(q2)
        if s[2] == "x":
            qc.h(q3)
        elif s[2] == "y":
            qc.y(q3)
        if s[3] == "x":
            qc.h(q4)
        elif s[3] == "y":
            qc.y(q4)
    else:
        raise ValueError("Not a valid Pauli gate")


def get_mixer_operator_circuit(G, beta):
    N = G.number_of_nodes()
    qc = QuantumCircuit(N)
    for n in G.nodes():
        append_x_term(qc, n, beta)
    return qc


def get_maxcut_cost_operator_circuit(G, gamma):
    N = G.number_of_nodes()
    qc = QuantumCircuit(N)
    for i, j in G.edges():
        if nx.is_weighted(G):
            append_zz_term(qc, i, j, gamma * G[i][j]["weight"])
        else:
            append_zz_term(qc, i, j, gamma)
    return qc


def get_tsp_cost_operator_circuit(G, gamma, encoding="onehot"):
    """
    Generates a circuit for the TSP phase unitary.

    Parameters
    ----------
    G : networkx.Graph
        Graph to solve MaxCut on
    gamma :
        QAOA parameter gamma
    encoding : string, default "onehot"
        Type of encoding for the city ordering

    Returns
    -------
    qc : qiskit.QuantumCircuit
        Quantum circuit implementing the TSP phase unitary
    """
    if encoding == "onehot":
        N = G.number_of_nodes()
        if not nx.is_weighted(G):
            raise ValueError("Provided graph is not weighted")
        qc = QuantumCircuit(N**2)
        for n in range(N): # cycle over all cities in the input ordering
            for u in range(N):
                for v in range(N): #road from city v to city u
                    q1 = (n*N + u) % (N**2)
                    q2 = ((n+1)*N + v) % (N**2)
                    qc.rzz(qc, q1, q2, gamma * G[u][v]["weight"]) #does this work if i,j not connected?
        return qc


def get_ordering_swap_partial_mixing_circuit(G, i, j, u, v, encoding="onehot"):
    """
    Generates an ordering swap partial mixer for the TSP mixing unitary.

    Parameters
    ----------
    G : networkx.Graph
        Graph to solve MaxCut on
    i, j :
        Positions in the ordering to be swapped
    u, v :
        Cities to be swapped
    encoding : string, default "onehot"
        Type of encoding for the city ordering

    Returns
    -------
    qc : qiskit.QuantumCircuit
        Quantum circuit implementing the TSP phase unitary
    """
    if encoding == "onehot":
        N = G.number_of_nodes()
        qc = QuantumCircuit(N**2)

        return qc


def get_tsp_mixer_operator_circuit(G, beta, encoding="onehot"):
    if encoding == "onehot":
        N = G.number_of_nodes()
        qc = QuantumCircuit(N**2)
        
        return qc


def get_maxcut_qaoa_circuit(
    G, beta, gamma, transpile_to_basis=True, save_state=True, qr=None, cr=None
):
    """Generates a circuit for weighted MaxCut on graph G.

    Parameters
    ----------
    G : networkx.Graph
        Graph to solve MaxCut on
    beta : list-like
        QAOA parameter beta
    gamma : list-like
        QAOA parameter gamma
    transpile_to_basis : bool, default True
        Transpile the circuit to ["u1", "u2", "u3", "cx"]
    save_state : bool, default True
        Add save state instruction to the end of the circuit
    qr : qiskit.QuantumRegister, default None
        Registers to use for the circuit.
        Useful when one has to compose circuits in a complicated way
        By default, G.number_of_nodes() registers are used
    cr : qiskit.ClassicalRegister, default None
        Classical registers, useful if measuring
        By default, no classical registers are added

    Returns
    -------
    qc : qiskit.QuantumCircuit
        Quantum circuit implementing QAOA
    """
    assert len(beta) == len(gamma)
    p = len(beta)  # infering number of QAOA steps from the parameters passed
    N = G.number_of_nodes()
    if qr is not None:
        assert isinstance(qr, QuantumRegister)
        assert qr.size >= N
    else:
        qr = QuantumRegister(N)

    if cr is not None:
        assert isinstance(cr, ClassicalRegister)
        qc = QuantumCircuit(qr, cr)
    else:
        qc = QuantumCircuit(qr)

    # first, apply a layer of Hadamards
    qc.h(range(N))
    # second, apply p alternating operators
    for i in range(p):
        qc = qc.compose(get_maxcut_cost_operator_circuit(G, gamma[i]))
        qc = qc.compose(get_mixer_operator_circuit(G, beta[i]))
    if transpile_to_basis:
        qc = transpile(qc, optimization_level=0, basis_gates=["u1", "u2", "u3", "cx"])
    if save_state:
        qc.save_state()
    return qc
