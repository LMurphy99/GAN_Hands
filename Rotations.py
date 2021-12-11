from math import atan, cos, sin, pi
import numpy as np
# np.set_printoptions(precision=2)
# np.set_printoptions(suppress=True)

def Rx(rad):
    return np.array([[1,0,0], [0, cos(rad), -sin(rad)], [0, sin(rad), cos(rad)]])

def Ry(rad):
    return np.array([[cos(rad), 0, sin(rad)], [0,1,0], [-sin(rad), 0, cos(rad)]])

def Rz(rad):
    return np.array([[cos(rad), -sin(rad), 0], [sin(rad), cos(rad), 0], [0,0,1]])

def findRotationXZY(a, b):
    """find rotation Rxz for a and subsequently Ry for b''.
    Returns Rxzy and theta, psi, and omega parameters."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    
    theta = atan( -a[2] / a[1] ) # rotation about x
    aX = np.matmul(Rx(theta), a)
    bX = np.matmul(Rx(theta), b)
    
    psi = atan( aX[0] / aX[1] ) # rotation about z
    #aXZ = np.matmul(Rz(psi), aX)
    bXZ = np.matmul(Rz(psi), bX)
    
    omega = atan( bXZ[2] / bXZ[0] ) # rotation about y
    bXZY = np.matmul(Ry(omega), bXZ)
    
    Rxz = np.matmul(Rz(psi), Rx(theta)) 
    
    if bXZY[0] > 0: # review this code
        return [np.matmul(Ry(omega), Rxz), theta, psi, omega]
    return [np.matmul(Ry(omega-pi), Rxz), theta, psi, omega-pi]