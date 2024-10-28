# Modified implementation of (C) 2017 Bhargav Chippada <bhargavchippada19@gmail.com>

from math import sqrt
# import numpy as np
from operator import add, sub
# This will substitute for the nLayout object
class Node:
    def __init__(self, dim):
        # self.mass = 0.0
        # self.old_dx = 0.0
        # self.old_dy = 0.0
        # self.dx = 0.0
        # self.dy = 0.0
        # self.x = 0.0
        # self.y = 0.0
        self.old_delta = [0.0] * dim
        self.delta = [0.0] * dim
        self.position = [0.0] * dim
        


# This is not in the original java code, but it makes it easier to deal with edges
class Edge:
    def __init__(self):
        self.node1 = -1
        self.node2 = -1
        self.weight = 0.0


# Here are some functions from ForceFactory.java
# =============================================

# Repulsion function.  `n1` and `n2` should be nodes.  This will
# adjust the dx and dy values of `n1`  `n2`
def linRepulsion(n1, n2, coefficient=0):
    # xDist = n1.x - n2.x
    # yDist = n1.y - n2.y
    # convert to numpy array
    n1_pos = n1.position
    n2_pos = n2.position
    displacement = [p1 - p2 for p1, p2 in zip(n1_pos, n2_pos)]
    distance2 = sum([d ** 2 for d in displacement])  # Distance squared

    if distance2 > 0:
        factor = coefficient * n1.mass * n2.mass / distance2
        # n1.dx += xDist * factor
        # n1.dy += yDist * factor
        # n2.dx -= xDist * factor
        # n2.dy -= yDist * factor
        n1.delta = list(map(add, n1.delta, [d * factor for d in displacement]))
        n2.delta = list(map(sub, n2.delta, [d * factor for d in displacement]))


# Repulsion function. 'n' is node and 'r' is region
def linRepulsion_region(n, r, coefficient=0):
    # xDist = n.x - r.massCenterX
    # yDist = n.y - r.massCenterY
    # distance2 = xDist * xDist + yDist * yDist
    position = n.position
    massCenter = r.massCenter
    displacement = list(map(sub, position, massCenter))
    # distance2 = displacement ** 2
    distance2 = sum([d ** 2 for d in displacement])

    if distance2 > 0:
        factor = coefficient * n.mass * r.mass / distance2
        # n.dx += xDist * factor
        # n.dy += yDist * factor
        # n.delta += displacement * factor
        n.delta = list(map(add, n.delta, [d * factor for d in displacement]))


# Gravity repulsion function.  For some reason, gravity was included
# within the linRepulsion function in the original gephi java code,
# which doesn't make any sense (considering a. gravity is unrelated to
# nodes repelling each other, and b. gravity is actually an
# attraction)
def linGravity(n, g):
    # xDist = n.x
    # yDist = n.y
    # distance = sqrt(xDist * xDist + yDist * yDist)
    displacement = n.position
    # distance = np.sqrt(np.sum(displacement ** 2))
    distance = sum([d ** 2 for d in displacement]) ** 0.5

    if distance > 0:
        factor = n.mass * g / distance
        # n.dx -= xDist * factor
        # n.dy -= yDist * factor
        # n.delta -= displacement * factor
        n.delta = list(map(sub, n.delta, [d * factor for d in displacement]))


# Strong gravity force function. `n` should be a node, and `g`
# should be a constant by which to apply the force.
def strongGravity(n, g, coefficient=0):
    # xDist = n.x
    # yDist = n.y
    # if xDist != 0 and yDist != 0:
    #     factor = coefficient * n.mass * g
    #     n.dx -= xDist * factor
    #     n.dy -= yDist * factor
    displacement = n.position
    # if np.all(displacement != 0):
    #     factor = coefficient * n.mass * g
    #     n.delta -= displacement * factor
    if not all(d == 0 for d in displacement):
        factor = coefficient * n.mass * g
        # n.delta -= displacement * factor
        n.delta = list(map(sub, n.delta, [d * factor for d in displacement]))



# Attraction function.  `n1` and `n2` should be nodes.  This will
# adjust the dx and dy values of `n1` and `n2`.  It does
# not return anything.
def linAttraction(n1, n2, e, distributedAttraction, coefficient=0):
    # xDist = n1.x - n2.x
    # yDist = n1.y - n2.y
    # displacement = n1.position - n2.position
    position1 = n1.position
    position2 = n2.position
    # displacement = position1 - position2
    displacement = list(map(sub, position1, position2))
    if not distributedAttraction:
        factor = -coefficient * e
    else:
        factor = -coefficient * e / n1.mass
    # n1.dx += xDist * factor
    # n1.dy += yDist * factor
    # n2.dx -= xDist * factor
    # n2.dy -= yDist * factor
    # n1.delta += displacement * factor
    # n2.delta -= displacement * factor
    n1.delta = list(map(add, n1.delta, [d * factor for d in displacement]))
    n2.delta = list(map(sub, n2.delta, [d * factor for d in displacement]))


# The following functions iterate through the nodes or edges and apply
# the forces directly to the node objects.  These iterations are here
# instead of the main file because Python is slow with loops.
def apply_repulsion(nodes, coefficient):
    i = 0
    for n1 in nodes:
        j = i
        for n2 in nodes:
            if j == 0:
                break
            linRepulsion(n1, n2, coefficient)
            j -= 1
        i += 1


def apply_gravity(nodes, gravity, scalingRatio, useStrongGravity=False):
    if not useStrongGravity:
        for n in nodes:
            linGravity(n, gravity)
    else:
        for n in nodes:
            strongGravity(n, gravity, scalingRatio)


def apply_attraction(nodes, edges, distributedAttraction, coefficient, edgeWeightInfluence):
    # Optimization, since usually edgeWeightInfluence is 0 or 1, and pow is slow
    if edgeWeightInfluence == 0:
        for edge in edges:
            linAttraction(nodes[edge.node1], nodes[edge.node2], 1, distributedAttraction, coefficient)
    elif edgeWeightInfluence == 1:
        for edge in edges:
            linAttraction(nodes[edge.node1], nodes[edge.node2], edge.weight, distributedAttraction, coefficient)
    else:
        for edge in edges:
            linAttraction(nodes[edge.node1], nodes[edge.node2], pow(edge.weight, edgeWeightInfluence),
                          distributedAttraction, coefficient)


# For Barnes Hut Optimization
class Region:
    def __init__(self, nodes, dim):
        self.mass = 0.0
        # self.massCenterX = 0.0
        # self.massCenterY = 0.0
        self.massCenter = [0.0] * dim
        self.size = 0.0
        self.nodes = nodes
        self.subregions = []
        self.updateMassAndGeometry()

    def updateMassAndGeometry(self):
        if len(self.nodes) > 1:
            self.mass = 0
            # massSumX = 0
            # massSumY = 0
            massSum = [0.0] * len(self.massCenter)
            for n in self.nodes:
                self.mass += n.mass
                # massSumX += n.x * n.mass
                # massSumY += n.y * n.mass
                position = n.position
                # print(np.asarray(n.position).shape)
                # massSum += position * n.mass
                massSum = list(map(add, massSum, [p * n.mass for p in position]))
            # self.massCenterX = massSumX / self.mass
            # self.massCenterY = massSumY / self.mass
            self.massCenter = [p / self.mass for p in massSum]

            self.size = 0.0
            for n in self.nodes:
                # distance = sqrt((n.x - self.massCenterX) ** 2 + (n.y - self.massCenterY) ** 2)
                position = n.position
                massCenter = self.massCenter
                # distance = np.sqrt(np.sum((position - massCenter) ** 2))
                distance = sum([(p - mc) ** 2 for p, mc in zip(position, massCenter)]) ** 0.5
                self.size = max(self.size, 2 * distance)

    def buildSubRegions(self):
        if len(self.nodes) > 1:
            # topleftNodes = []
            # bottomleftNodes = []
            # toprightNodes = []
            # bottomrightNodes = []

            # 2 ^ dim subregions
            numSubregions = int(2 ** len(self.massCenter))
            subregions = [[] for _ in range(numSubregions)]
            # Optimization: The distribution of self.nodes into 
            # subregions now requires only one for loop. Removed 
            # topNodes and bottomNodes arrays: memory space saving.
            for n in self.nodes:
                # partition into hypercubes centered at self.massCenter
                subregionIndex = 0
                for i in range(len(self.massCenter)):
                    if n.position[i] > self.massCenter[i]:
                        subregionIndex += 2 ** i
                subregions[subregionIndex].append(n)
                # if n.x < self.massCenterX:
                #     if n.y < self.massCenterY:
                #         bottomleftNodes.append(n)
                #     else:
                #         topleftNodes.append(n)
                # else:
                #     if n.y < self.massCenterY:
                #         bottomrightNodes.append(n)
                #     else:
                #         toprightNodes.append(n)      
            for subregionNodes in subregions:
                if len(subregionNodes) > 0:
                    if len(subregionNodes) < len(self.nodes):
                        subregion = Region(subregionNodes, dim=len(self.massCenter))
                        self.subregions.append(subregion)
                    else:
                        for n in subregionNodes:
                            subregion = Region([n], dim=len(self.massCenter))
                            self.subregions.append(subregion)

            # if len(topleftNodes) > 0:
            #     if len(topleftNodes) < len(self.nodes):
            #         subregion = Region(topleftNodes)
            #         self.subregions.append(subregion)
            #     else:
            #         for n in topleftNodes:
            #             subregion = Region([n])
            #             self.subregions.append(subregion)

            # if len(bottomleftNodes) > 0:
            #     if len(bottomleftNodes) < len(self.nodes):
            #         subregion = Region(bottomleftNodes)
            #         self.subregions.append(subregion)
            #     else:
            #         for n in bottomleftNodes:
            #             subregion = Region([n])
            #             self.subregions.append(subregion)

            # if len(toprightNodes) > 0:
            #     if len(toprightNodes) < len(self.nodes):
            #         subregion = Region(toprightNodes)
            #         self.subregions.append(subregion)
            #     else:
            #         for n in toprightNodes:
            #             subregion = Region([n])
            #             self.subregions.append(subregion)

            # if len(bottomrightNodes) > 0:
            #     if len(bottomrightNodes) < len(self.nodes):
            #         subregion = Region(bottomrightNodes)
            #         self.subregions.append(subregion)
            #     else:
            #         for n in bottomrightNodes:
            #             subregion = Region([n])
            #             self.subregions.append(subregion)

            for subregion in self.subregions:
                subregion.buildSubRegions()

    def applyForce(self, n, theta, coefficient=0):
        if len(self.nodes) < 2:
            linRepulsion(n, self.nodes[0], coefficient)
        else:
            # distance = sqrt((n.x - self.massCenterX) ** 2 + (n.y - self.massCenterY) ** 2)
            position = n.position
            massCenter = self.massCenter
            # distance = np.sqrt(np.sum((position - massCenter) ** 2))
            distance = sum([(p - mc) ** 2 for p, mc in zip(position, massCenter)] ) ** 0.5
            # distance = np.sqrt(np.sum((n.position - self.massCenter) ** 2))
            if distance * theta > self.size:
                linRepulsion_region(n, self, coefficient)
            else:
                for subregion in self.subregions:
                    subregion.applyForce(n, theta, coefficient)

    def applyForceOnNodes(self, nodes, theta, coefficient=0):
        for n in nodes:
            self.applyForce(n, theta, coefficient)


# Adjust speed and apply forces step
def adjustSpeedAndApplyForces(nodes, speed, speedEfficiency, jitterTolerance):
    # Auto adjust speed.
    totalSwinging = 0.0  # How much irregular movement
    totalEffectiveTraction = 0.0  # How much useful movement
    for n in nodes:
        # swinging = sqrt((n.old_dx - n.dx) * (n.old_dx - n.dx) + (n.old_dy - n.dy) * (n.old_dy - n.dy))
        old_delta = n.old_delta
        delta = n.delta
        # swinging = np.sqrt(np.sum((n.old_delta - n.delta) ** 2))
        # swinging = np.sqrt(np.sum((old_delta - delta) ** 2))
        swinging = sum([(od - d) ** 2 for od, d in zip(old_delta, delta)]) ** 0.5
        totalSwinging += n.mass * swinging
        # totalEffectiveTraction += .5 * n.mass * sqrt(
            # (n.old_dx + n.dx) * (n.old_dx + n.dx) + (n.old_dy + n.dy) * (n.old_dy + n.dy))
        # totalEffectiveTraction += .5 * n.mass * np.sqrt(np.sum((n.old_delta + n.delta) ** 2))
        # totalEffectiveTraction += .5 * n.mass * np.sqrt(np.sum((old_delta + delta) ** 2))
        totalEffectiveTraction += .5 * n.mass * sum([(od + d) ** 2 for od, d in zip(old_delta, delta)]) ** 0.5
    # Optimize jitter tolerance.  The 'right' jitter tolerance for
    # this network. Bigger networks need more tolerance. Denser
    # networks need less tolerance. Totally empiric.
    estimatedOptimalJitterTolerance = .05 * sqrt(len(nodes))
    minJT = sqrt(estimatedOptimalJitterTolerance)
    maxJT = 10
    jt = jitterTolerance * max(minJT,
                               min(maxJT, estimatedOptimalJitterTolerance * totalEffectiveTraction / (
                                   len(nodes) * len(nodes))))

    minSpeedEfficiency = 0.05

    # Protective against erratic behavior
    if totalEffectiveTraction and totalSwinging / totalEffectiveTraction > 2.0:
        if speedEfficiency > minSpeedEfficiency:
            speedEfficiency *= .5
        jt = max(jt, jitterTolerance)

    if totalSwinging == 0:
        targetSpeed = float('inf')
    else:
        targetSpeed = jt * speedEfficiency * totalEffectiveTraction / totalSwinging

    if totalSwinging > jt * totalEffectiveTraction:
        if speedEfficiency > minSpeedEfficiency:
            speedEfficiency *= .7
    elif speed < 1000:
        speedEfficiency *= 1.3

    # But the speed shoudn't rise too much too quickly, since it would
    # make the convergence drop dramatically.
    maxRise = .5
    speed = speed + min(targetSpeed - speed, maxRise * speed)

    # Apply forces.
    #
    # Need to add a case if adjustSizes ("prevent overlap") is
    # implemented.
    for n in nodes:
        # swinging = n.mass * sqrt((n.old_dx - n.dx) * (n.old_dx - n.dx) + (n.old_dy - n.dy) * (n.old_dy - n.dy))
        # swinging = n.mass * np.sqrt(np.sum((n.old_delta - n.delta) ** 2))
        old_delta = n.old_delta
        delta = n.delta
        # swinging = n.mass * np.sqrt(np.sum((old_delta - delta) ** 2))
        swinging = n.mass * sum([(od - d) ** 2 for od, d in zip(old_delta, delta)]) ** 0.5
        factor = speed / (1.0 + sqrt(speed * swinging))
        # n.x = n.x + (n.dx * factor)
        # n.y = n.y + (n.dy * factor)

        # n.position = n.position + (n.delta * factor)
        position = n.position
        # print("before: ", np.asarray(n.position).shape)
        # n.position = position + (delta * factor)
        n.position = list(map(add, position, [d * factor for d in delta]))
        # print("after: ", np.asarray(n.position).shape)
        # print()
        # print()

    values = {}
    values['speed'] = speed
    values['speedEfficiency'] = speedEfficiency

    return values


try:
    import cython

    if not cython.compiled:
        print("Warning: uncompiled fa2util module.  Compile with cython for a 10-100x speed boost.")
except:
    print("No cython detected.  Install cython and compile the fa2util module for a 10-100x speed boost.")