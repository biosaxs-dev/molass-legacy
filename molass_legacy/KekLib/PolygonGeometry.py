"""
    PolygonGeometry.py

    borrowed from
    https://stackoverflow.com/questions/68104969/offset-a-parallel-line-to-a-given-line-python/68109283    
"""
import numpy as np

def normalizeVec(x,y):
    distance = np.sqrt(x*x+y*y)
    return x/distance, y/distance

def makeOffsetPoly(oldX, oldY, offset, outer_ccw = 1):
    """
    there seems to be a bug when outer_ccw = -1 for inner offset
    use a negetive offset instead
    """
    num_points = len(oldX)
    newX = []
    newY = []

    for curr in range(num_points):
        prev = (curr + num_points - 1) % num_points
        next = (curr + 1) % num_points

        vnX =  oldX[next] - oldX[curr]
        vnY =  oldY[next] - oldY[curr]
        vnnX, vnnY = normalizeVec(vnX,vnY)
        nnnX = vnnY
        nnnY = -vnnX

        vpX =  oldX[curr] - oldX[prev]
        vpY =  oldY[curr] - oldY[prev]
        vpnX, vpnY = normalizeVec(vpX,vpY)
        npnX = vpnY * outer_ccw
        npnY = -vpnX * outer_ccw

        bisX = (nnnX + npnX) * outer_ccw
        bisY = (nnnY + npnY) * outer_ccw

        bisnX, bisnY = normalizeVec(bisX,  bisY)
        bislen = offset /  np.sqrt((1 + nnnX*npnX + nnnY*npnY)/2)

        newX.append(oldX[curr] + bislen * bisnX)
        newY.append(oldY[curr] + bislen * bisnY)
    
    return newX, newY