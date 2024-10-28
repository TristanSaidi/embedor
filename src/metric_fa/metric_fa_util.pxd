import cython
from cython.view cimport array as cvarray
cimport numpy as np
# np.import_array()

cdef class Node:
    cdef public double mass  # Define mass as a double for efficient storage
    # cdef public double[:] old_delta  # 1D array of doubles
    # cdef public double[:] delta      # 1D array of doubles
    # cdef public double[:] position   # 1D array of doubles
    cdef public list old_delta
    cdef public list delta
    cdef public list position

    # cdef inline __cinit__(self, int dim):
    #     # Initialize each attribute, ensuring arrays are created with the specified dimension
    #     self.mass = 0.0
    #     self.old_delta = np.zeros(dim, dtype=np.float64)
    #     self.delta = np.zeros(dim, dtype=np.float64)
    #     self.position = np.zeros(dim, dtype=np.float64)


# This is not in the original java function, but it makes it easier to
# deal with edges.
cdef class Edge:
    cdef public int node1, node2
    cdef public double weight

# Repulsion function.  `n1` and `n2` should be nodes.  This will
# adjust the dx and dy values of `n1` (and optionally `n2`).  It does
# not return anything.

@cython.locals(displacement = list,
               distance2 = cython.double, 
               factor = cython.double)
cdef void linRepulsion(Node n1, Node n2, double coefficient=*)

@cython.locals(displacement = list,
               distance2 = cython.double,
               factor = cython.double)
cdef void linRepulsion_region(Node n, Region r, double coefficient=*)


@cython.locals(displacement = list,
               distance = cython.double, 
               factor = cython.double)
cdef void linGravity(Node n, double g)


@cython.locals(displacement = list,
               factor = cython.double)
cdef void strongGravity(Node n, double g, double coefficient=*)

@cython.locals(displacement = list,
               factor = cython.double)
cpdef void linAttraction(Node n1, Node n2, double e, bint distributedAttraction, double coefficient=*)

@cython.locals(i = cython.int,
               j = cython.int,
               n1 = Node,
               n2 = Node)
cpdef void apply_repulsion(list nodes, double coefficient)

@cython.locals(n = Node)
cpdef void apply_gravity(list nodes, double gravity, double scalingRatio, bint useStrongGravity=*)

@cython.locals(edge = Edge)
cpdef void apply_attraction(list nodes, list edges, bint distributedAttraction, double coefficient, double edgeWeightInfluence)

cdef class Region:
    cdef double mass
    cdef list massCenter  # 1D array for mass center in given dimensions
    cdef double size
    cdef object nodes  # For a list of nodes, we’ll use Python’s dynamic typing
    cdef list subregions  # List of subregions as a standard Python list

    # cdef inline __cinit__(self, nodes, int dim):
    #     self.mass = 0.0
    #     self.massCenter = np.zeros(dim, dtype=np.float64)
    #     self.size = 0.0
    #     self.nodes = nodes
    #     self.subregions = []

    @cython.locals(massSum = list,
                   position = list,
                   massCenter = list,
                   n = Node,
                   distance = cython.double)
    cdef void updateMassAndGeometry(self)

    @cython.locals(n = Node,
                   numSubregions = int,
                   subregions = list,
                   subregion = Region)
    cpdef void buildSubRegions(self)


    @cython.locals(distance = cython.double,
                   subregion = Region)
    cdef void applyForce(self, Node n, double theta, double coefficient=*)

    @cython.locals(n = Node)
    cpdef applyForceOnNodes(self, list nodes, double theta, double coefficient=*)

@cython.locals(totalSwinging = cython.double,
               totalEffectiveTraction = cython.double,
               n = Node,
               swinging = cython.double,
               totalSwinging = cython.double,
               totalEffectiveTraction = cython.double,
               estimatedOptimalJitterTolerance = cython.double,
               minJT = cython.double,
               maxJT = cython.double,
               jt = cython.double,
               minSpeedEfficiency = cython.double,
               targetSpeed = cython.double,
               maxRise = cython.double,
               factor = cython.double,
               values = dict)
cpdef dict adjustSpeedAndApplyForces(list nodes, double speed, double speedEfficiency, double jitterTolerance)