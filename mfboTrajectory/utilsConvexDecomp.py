#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os, sys, time, copy
import yaml
import matplotlib.pyplot as plt

class Graph():
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)] 
                    for row in range(vertices)]
        return
 
    # A utility function to print the constructed MST stored in parent[]
    def printMST(self, parent):
        print("Edge \tWeight")
        for i in range(1, self.V):
            print(parent[i], "-", i, "\t", self.graph[i][parent[i]])
        return
 
    # A utility function to find the vertex with 
    # minimum distance value, from the set of vertices 
    # not yet included in shortest path tree
    def minKey(self, key, mstSet):
        min_t = np.iinfo(np.int32).max
        for v in range(self.V):
            if key[v] < min_t and mstSet[v] == False:
                min_t = key[v]
                min_index = v
 
        return min_index

    def primMST(self):
        # Key values used to pick minimum weight edge in cut
        key = [np.iinfo(np.int32).max] * self.V
        parent = [None] * self.V # Array to store constructed MST
        # Make key 0 so that this vertex is picked as first vertex
        key[0] = 0
        mstSet = [False] * self.V
 
        parent[0] = -1 # First node is always the root of
 
        for cout in range(self.V):
            u = self.minKey(key, mstSet)
            mstSet[u] = True
            for v in range(self.V):
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
                        key[v] = self.graph[u][v]
                        parent[v] = u
        
#         self.printMST(parent)
        return parent
   
    def printSolution(self, dist): 
        print ("Vertex tDistance from Source") 
        for node in range(self.V): 
            print (node, "t", dist[node]) 
        return
   
    def minDistance(self, dist, sptSet): 
        # Initilaize minimum distance for next node 
        min_t = np.iinfo(np.int32).max
        min_index = 0
        for v in range(self.V):
            if dist[v] < min_t and sptSet[v] == False: 
                min_t = dist[v] 
                min_index = v 
   
        return min_index 

    def dijkstra(self, src): 
        dist = [sys.maxsize] * self.V 
        dist[src] = 0
        sptSet = [False] * self.V 
        parent = [-1] * self.V 
        for cout in range(self.V): 
            u = self.minDistance(dist, sptSet) 
            sptSet[u] = True

            for v in range(self.V):
                if (self.graph[u][v] > 0) and sptSet[v] == False and dist[v] > dist[u] + self.graph[u][v]: 
                    dist[v] = dist[u] + self.graph[u][v]
                    parent[v] = u 
#         self.printSolution(dist)
        return dist, parent

class _Point():
    def __init__(self,x=0,y=0,idx=[0,0]):
        self.idx = idx
        self.x = x
        self.y = y
        
    def __str__(self):
        return "[%s, %s]\n" % (self.x, self.y)
    
    def __add__(self, o):
        return _Point(self.x+o.x, self.y+o.y, self.idx)
    
    def __sub__(self, o):
        return _Point(self.x-o.x, self.y-o.y, self.idx)

    def __neg__(self):
        return _Point(-self.x, -self.y, self.idx)
    
    def __mul__(self, h):
        return _Point(self.x*h, self.y*h, self.idx)
    
    def normalize(self):
        norm = np.sqrt(self.x**2+self.y**2)
        
        if norm > 1e-3:
            return _Point(self.x/norm, self.y/norm, self.idx)
        else:
            return _Point(0,0,self.idx)

# Given three colinear points p, q, r, the function checks if  
# point q lies on line segment 'pr'  
def onSegment(p, q, r): 
    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
           (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))): 
        return True
    return False
  
def orientation(p, q, r): 
    # to find the orientation of an ordered triplet (p,q,r) 
    # function returns the following values: 
    # 0 : Colinear points 
    # 1 : Clockwise points 
    # 2 : Counterclockwise 
      
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/  
    # for details of below formula.  
    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y)) 
    if (val > 0): 
        # Clockwise orientation 
        return 1
    elif (val < 0):
        # Counterclockwise orientation 
        return 2
    else: 
        # Colinear orientation 
        return 0

# The main function that returns true if  
# the line segment 'p1q1' and 'p2q2' intersect. 
def checkIntersect(p1,q1,p2,q2): 
#     i = _Point()
#     a1 = q1.y - p1.y
#     b1 = p1.x - q1.x
#     c1 = a1*p1.x + b1*p1.y
#     a2 = q2.y - p2.y
#     b2 = p2.x - q2.x
#     c2 = a2*p2.x + b2*p2.y
#     det = a1*b2 - a2*b1
#     if det < 1e-3 and det > -1e-3:
#         return False
#     else:
#         i.x = (b2 * c1 - b1 * c2) / det
#         i.y = (a1 * c2 - a2 * c1) / det
#         if sqdist(p1, i) > sqdist(p1,q1) or sqdist(q1, i) > sqdist(p1,q1):
#             return False
#         if sqdist(p2, i) > sqdist(p2,q2) or sqdist(q2, i) > sqdist(p2,q2):
#             return False
        
#         return True
      
    # Find the 4 orientations required for  
    # the general and special cases 
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 
  
    # General case 
    if ((o1 != o2) and (o3 != o4)): 
        return True
  
    # Special Cases 
    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1 
    if ((o1 == 0) and onSegment(p1, p2, q1)): 
        return True
  
    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1 
    if ((o2 == 0) and onSegment(p1, q2, q1)): 
        return True
  
    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2 
    if ((o3 == 0) and onSegment(p2, p1, q2)): 
        return True
  
    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2 
    if ((o4 == 0) and onSegment(p2, q1, q2)): 
        return True
  
    # If none of the cases 
    return False

class _Line():
    def __init__(self,p1,p2):
        self.first = p1
        self.second = p2
        
    def __str__(self):
        return self.first.__str__() + self.second.__str__()

def area(a,b,c):
    return (b.x-a.x)*(c.y-a.y)-(c.x-a.x)*(b.y-a.y)

def left(a,b,c):
    return area(a,b,c) > 0

def leftOn(a,b,c):
    return area(a,b,c) >= 0

def right(a,b,c):
    return area(a,b,c) < 0

def rightOn(a,b,c):
    return area(a,b,c) <= 0

def collinear(a,b,c):
    return area(a,b,c) == 0

def sqdist(a,b):
    return (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y)

# Line intersection
def lineInt(l1,l2):
    i = _Point()
    a1 = l1.second.y - l1.first.y
    b1 = l1.first.x - l1.second.x
    c1 = a1*l1.first.x + b1*l1.first.y
    a2 = l2.second.y - l2.first.y
    b2 = l2.first.x - l2.second.x
    c2 = a2*l2.first.x + b2*l2.first.y
    det = a1*b2 - a2*b1
    if det < 1e-3 and det > -1e-3:
        return None
    else:
        i.x = (b2 * c1 - b1 * c2) / det
        i.y = (a1 * c2 - a2 * c1) / det
        if sqdist(l1.first, i) > sqdist(l1.first, l1.second) or sqdist(l1.second, i) > sqdist(l1.first, l1.second):
            return None
        if sqdist(l2.first, i) > sqdist(l2.first, l2.second) or sqdist(l2.second, i) > sqdist(l2.first, l2.second):
            return None
        
        return i

# points_set: outer polygon + inner holes
class _Polygon:
    def __init__(self, points_set=[]):
        self.bridge_eps = 0.01
#         self.bridge_eps = 0.1
        self.points = []
        self.points_original = []
        self.points_set = points_set
        self.bridge = None

        if len(points_set) > 1 :
            points_init = []
            for i in range(len(points_set)):
                points_init_t = []
                for j in range(len(points_set[i])):
                    points_init_t.append(_Point(points_set[i][j][0],points_set[i][j][1],[i,j]))
                if i==0:
                    points_init_t = self.makeCCW(points_init_t)
                else:
                    points_init_t = self.makeCW(points_init_t)
                points_init.append(points_init_t)
            parent_idx, bridge_info = self.find_bridge_all(points_init)
            self.bridge = [parent_idx, bridge_info]

            def add_points(node_idx, start_idx=0, prev_point=None):
                # Check child node and the index of corresponding output vertex
                child_idx = []
                for i in range(1,len(parent_idx)):
                    if parent_idx[i] == node_idx:
                        child_idx.append(i)
                child_info = []
                out_vertex_idx = []
                for i in range(len(child_idx)):
                    child_info.append([bridge_info[node_idx,child_idx[i],:],child_idx[i]])
                child_info.sort(key=(lambda x: x[0][1]))
                for i in range(len(child_info)):
                    out_vertex_idx.append(child_info[i][0][1])
                
                if np.any(prev_point==None):
                    prev_point_t0 = points_set[node_idx][(start_idx-1)%len(points_set[node_idx])]
                    prev_point_t = _Point(prev_point_t0[0], prev_point_t0[1])
                    current_polygon_vertex_idx = list(range(start_idx,len(points_set[node_idx])))+list(range(0,start_idx))
                else:
                    prev_point_t = prev_point
                    prev_point_original = prev_point
                    current_polygon_vertex_idx = list(range(start_idx,len(points_set[node_idx])))+list(range(0,start_idx+1))
                
                next_point_idx = len(self.points)
                flag_first = True
                for i in current_polygon_vertex_idx:
                    self.points.append(_Point(points_set[node_idx][i][0],points_set[node_idx][i][1],[node_idx,i]))
                    out_vertex_idx_match = [i_t for i_t, x in enumerate(out_vertex_idx) if x == i]
                    for v_idx in out_vertex_idx_match:
                        p_idx_i = len(self.points)-1
                        curr_point_t = self.points[-1]
                        next_point_t = add_points(\
                            node_idx=child_info[v_idx][1], \
                            start_idx=np.int(child_info[v_idx][0][2]), \
                            prev_point=curr_point_t)
                        v_o_i = (-(curr_point_t-prev_point_t).normalize() \
                                 -(curr_point_t-next_point_t).normalize()).normalize()*self.bridge_eps
                        # Check couter clockwise
                        if right(prev_point_t, curr_point_t, next_point_t):
                            v_o_i_sign = -1
                        else:
                            v_o_i_sign = 1
                        if v_o_i.x == 0 and v_o_i.y==0:
                            v_o_i.x = -(curr_point_t-prev_point_t).y
                            v_o_i.y = (curr_point_t-prev_point_t).x
                            v_o_i = v_o_i.normalize()*self.bridge_eps
                        self.points[p_idx_i] = self.points[p_idx_i]+v_o_i*v_o_i_sign
                        self.points.append(_Point(points_set[node_idx][i][0],points_set[node_idx][i][1],[node_idx,i]))
                        prev_point_t = self.points[-1]
                    if len(out_vertex_idx_match) == 0 and flag_first and not np.any(prev_point==None):
                        curr_point_t = self.points[-1]
                        next_point_idx2 = (i+1)%len(points_set[node_idx])
                        next_point_t = _Point(points_set[node_idx][next_point_idx2][0], \
                                              points_set[node_idx][next_point_idx2][1])
                        v_o_i = (-(curr_point_t-prev_point_t).normalize() \
                                 -(curr_point_t-next_point_t).normalize()).normalize()*self.bridge_eps
                        # Check couter clockwise
                        if right(prev_point_t, curr_point_t, next_point_t):
                            v_o_i_sign = -1
                        else:
                            v_o_i_sign = 1
                        if v_o_i.x == 0 and v_o_i.y==0:
                            v_o_i.x = -(curr_point_t-prev_point_t).y
                            v_o_i.y = (curr_point_t-prev_point_t).x
                            v_o_i = v_o_i.normalize()*self.bridge_eps
                        self.points[-1] = self.points[-1]+v_o_i*v_o_i_sign
                    if len(out_vertex_idx_match) > 0:
                        curr_point_t = self.points[-1]
                        next_point_idx2 = (i+1)%len(points_set[node_idx])
                        next_point_t2 = _Point(points_set[node_idx][next_point_idx2][0], \
                                              points_set[node_idx][next_point_idx2][1])
                        v_o_i = (-(curr_point_t-next_point_t).normalize() \
                                 -(curr_point_t-next_point_t2).normalize()).normalize()*self.bridge_eps
                        # Check couter clockwise
                        if right(next_point_t, curr_point_t, next_point_t2):
                            v_o_i_sign = -1
                        else:
                            v_o_i_sign = 1
                        if v_o_i.x == 0 and v_o_i.y==0:
                            v_o_i.x = -(curr_point_t-next_point_t).y
                            v_o_i.y = (curr_point_t-next_point_t).x
                            v_o_i = v_o_i.normalize()*self.bridge_eps
                        self.points[-1] = self.points[-1]+v_o_i*v_o_i_sign
                    flag_first = False
                    prev_point_t = self.points[-1]
                
                if not np.any(prev_point==None):
                    curr_point_t = self.points[-1]
                    next_point_idx2 = (i-1)%len(points_set[node_idx])
                    next_point_t = _Point(points_set[node_idx][next_point_idx2][0], \
                                          points_set[node_idx][next_point_idx2][1])
                    v_o_i = (-(curr_point_t-prev_point_original).normalize() \
                             -(curr_point_t-next_point_t).normalize()).normalize()*self.bridge_eps
                    
                    # Check couter clockwise
                    if right(next_point_t, curr_point_t, prev_point_original):
                        v_o_i_sign = -1
                    else:
                        v_o_i_sign = 1
                    self.points[-1] = self.points[-1]+v_o_i*v_o_i_sign
                return self.points[next_point_idx]
            
            add_points(0, start_idx=0, prev_point=None)
        
        elif len(points_set) == 1 :
            for i in range(len(points_set[0])):
                self.points.append(_Point(points_set[0][i][0],points_set[0][i][1],[0,i]))

        return
    
    def getIndexSet(self):
        idx_set = []
        for p in self.points:
            idx_set.append(p.idx)
        return idx_set
    
    def compareHistory(self, history, history_set):
        for i in range(len(history)):
            history_t = history[i:] + history[:i]
            if history_t in history_set:
                return True, history_set.index(history_t)
        return False, None
    
    def canSee(self, a, b):
        dist = 0
        if leftOn(self.points[(a+1)%len(self.points)], self.points[a], self.points[b]) \
            and rightOn(self.points[(a-1)%len(self.points)], self.points[a], self.points[b]):
            return False
        dist = sqdist(self.points[a], self.points[b])
        for i in range(len(self.points)):
            if ((i+1)%len(self.points) == a or i == a): # ignore incident edges
                continue
            if ((i+1)%len(self.points) == b or i == b): # ignore incident edges
                continue
            if leftOn(self.points[a], self.points[b], self.points[(i+1)%len(self.points)]) \
                and rightOn(self.points[a], self.points[b], self.points[i]):
#                 print("      -- canSeeCheck a={}, b={}, i={}".format(a,b,i))
                p = lineInt(_Line(self.points[a], self.points[b]), _Line(self.points[i], self.points[(i+1)%len(self.points)]))
                if p is None:
                    continue
                if (sqdist(self.points[a], p) < dist): # if edge is blocking visibility to b
#                     print("        -- False ",sqdist(self.points[a], p),dist)
                    return False
        return True

    def copy(self, i, j):
        p = _Polygon()
        p.points_set = copy.deepcopy(self.points_set)
        if (i < j):
            for k in range(i,j+1):
                p.points.append(_Point(self.points[k].x,self.points[k].y,self.points[k].idx))
        else:
            for k in range(i,len(self.points)):
                p.points.append(_Point(self.points[k].x,self.points[k].y,self.points[k].idx))
            for k in range(j+1):
                p.points.append(_Point(self.points[k].x,self.points[k].y,self.points[k].idx))
        return p

#     def copy2(self, i1, i2, j1, j2):
#         p1 = self.copy(i1,i2)
#         p2 = self.copy(j1,j2)
        
#         p = _Polygon()
#         p.points_set = copy.deepcopy(self.points_set)
#         for k in range(len(p1.points)):
#             p.points.append(p1.points[k])
#         for k in range(len(p2.points)):
#             p.points.append(p2.points[k])
#         return p
    
    def copy2(self, cut):
        p = _Polygon()
        p.points_set = copy.deepcopy(self.points_set)
        
        for idx, c in enumerate(cut):
            idx_n = (idx+1)%len(cut)
            p_t = self.copy(c[1],cut[idx_n][0])
            for k in range(len(p_t.points)):
                p.points.append(p_t.points[k])
        return p
    
    def decomp(self, manual_cut=[], search_history=[[],[]]):
        edge_min = []
        search_history_t = search_history
        ret, idx_t = self.compareHistory(self.getIndexSet(), search_history_t[0])
        if ret:
            return search_history_t[1][idx_t]
        
        nDiags = np.iinfo(np.int32).max
        
        if len(manual_cut) > 0:
            for i in range(len(manual_cut)):
                poly_t = self.copy2(manual_cut[i])
                poly_t.plot_approx()
                edge_min_t = poly_t.decomp(search_history=search_history_t)
                search_history_t[0].append(poly_t.getIndexSet())
                search_history_t[1].append(edge_min_t)
                for e in edge_min_t:
                    edge_min.append(e)
                for c in manual_cut[i]:
                    edge_min.append([self.points[c[0]].idx,self.points[c[1]].idx])
        else:
            for i in range(len(self.points)):
                if right(self.points[(i-1)%len(self.points)], self.points[i], self.points[(i+1)%len(self.points)]):
                    for j in range(len(self.points)):
                        if (self.canSee(i,j)):
                            poly_t1 = self.copy(i,j)
                            poly_t2 = self.copy(j,i)
                            edge_tmp1 = poly_t1.decomp(search_history=search_history_t)
                            search_history_t[0].append(poly_t1.getIndexSet())
                            search_history_t[1].append(edge_tmp1)
                            edge_tmp2 = poly_t2.decomp(search_history=search_history_t)
                            search_history_t[0].append(poly_t2.getIndexSet())
                            search_history_t[1].append(edge_tmp2)
                            
                            for e in edge_tmp2:
                                edge_tmp1.append(e)
                            if len(edge_tmp1) < nDiags:
                                edge_min = []
                                for e in edge_tmp1:
                                    edge_min.append(e)
                                nDiags = len(edge_tmp1)
                                edge_min.append([self.points[i].idx,self.points[j].idx])
        return edge_min
        
    def makeCCW(self, points_t):
        br = 0
        # find bottom right point
        for i in range(len(points_t)):
            if (points_t[i].y < points_t[br].y \
                or (points_t[i].y == points_t[br].y \
                    and points_t[i].x > points_t[br].x)):
                br = i

        # reverse poly if clockwise
        if not left(points_t[(br-1)%len(points_t)], points_t[br], points_t[(br+1)%len(points_t)]):
            points_t.reverse()
            for i in range(len(points_t)):
                points_t[i].idx = i
        return points_t
    
    def makeCW(self, points_t):
        tl = 0
        # find top left point
        for i in range(len(points_t)):
            if (points_t[i].y > points_t[tl].y \
                or (points_t[i].y == points_t[tl].y \
                    and points_t[i].x < points_t[tl].x)):
                tl = i

        # reverse poly if couter clockwise
        if not right(points_t[(tl-1)%len(points_t)], points_t[tl], points_t[(tl+1)%len(points_t)]):
            points_t.reverse()
            for i in range(len(points_t)):
                points_t[i].idx = i
        return points_t
    
    def find_bridge(self, points1, points2):
        min_dist = np.iinfo(np.int32).max
        idx1 = 0
        idx2 = 0
        for i in range(len(points1)):
            for j in range(len(points2)):
                if min_dist > np.sqrt(sqdist(points1[i], points2[j])):
                    min_dist = np.sqrt(sqdist(points1[i], points2[j]))
                    idx1 = i
                    idx2 = j
        return idx1, idx2, min_dist
    
    def find_bridge_all(self, points_set):
        # generate dist info
        dist_info = np.zeros((len(points_set),len(points_set),3))
        dist_mat = np.zeros((len(points_set),len(points_set)))
        for i in range(len(points_set)-1):
            for j in range(i+1,len(points_set)):
                idx1, idx2, min_dist = self.find_bridge(points_set[i],points_set[j])
                dist_info[i,j,:] = np.array([min_dist,idx1,idx2])
                dist_info[j,i,:] = np.array([min_dist,idx2,idx1])
                dist_mat[i,j] = min_dist
                dist_mat[j,i] = min_dist
        # find minimum spanning tree
        g = Graph(len(points_set))
        g.graph = dist_mat
        parent = g.primMST()
        return parent, dist_info
    
    def plot(self, decomp_edge=[]):
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        
        for i in range(len(self.points_set)):
            point_x = np.zeros(len(self.points_set[i])+1)
            point_y = np.zeros(len(self.points_set[i])+1)
            for j in range(len(self.points_set[i])):
                point_x[j] = self.points_set[i][j][0]
                point_y[j] = self.points_set[i][j][1]
            point_x[-1] = self.points_set[i][0][0]
            point_y[-1] = self.points_set[i][0][1]
            if i == 0:
                ax.plot(point_x, point_y, '-', linewidth=5, color='black')
            else:
                ax.plot(point_x, point_y, '-', linewidth=5, color='black')
                ax.fill(point_x, point_y, color='black', alpha=0.2)
        
        if self.bridge is not None:
            for i in range(1,len(self.bridge[0])):
                parent_idx = self.bridge[0][i]
                point_x = np.zeros(2)
                point_y = np.zeros(2)
                point_x[0] = self.points_set[parent_idx][np.int(self.bridge[1][parent_idx,i,1])][0]
                point_y[0] = self.points_set[parent_idx][np.int(self.bridge[1][parent_idx,i,1])][1]
                point_x[1] = self.points_set[i][np.int(self.bridge[1][parent_idx,i,2])][0]
                point_y[1] = self.points_set[i][np.int(self.bridge[1][parent_idx,i,2])][1]
                ax.plot(point_x, point_y, '-', color='tab:red')
            
        for i in range(len(decomp_edge)):
            point_x = np.zeros(2)
            point_y = np.zeros(2)
            point_x[0] = self.points_set[decomp_edge[i][0][0]][decomp_edge[i][0][1]][0]
            point_y[0] = self.points_set[decomp_edge[i][0][0]][decomp_edge[i][0][1]][1]
            point_x[1] = self.points_set[decomp_edge[i][1][0]][decomp_edge[i][1][1]][0]
            point_y[1] = self.points_set[decomp_edge[i][1][0]][decomp_edge[i][1][1]][1]
            ax.plot(point_x, point_y, '-', color='tab:orange')
        
        plt.grid()
        plt.show()
        plt.pause(0.1)
        return
    
    def save_decomp(self, decomp_edge, filedir='poly_decomp', filename='result.yaml'):
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        yamlFile = os.path.join(filedir, filename)
        yaml_out = open(yamlFile,"w")
        yaml_out.write("points:\n")
        for i in range(len(self.points_set)):
            yaml_out.write("    {}:\n".format(i))
            for j in range(len(self.points_set[i])):
                yaml_out.write("      - [{}, {}]\n".format(self.points_set[i][j][0],self.points_set[i][j][1]))
        yaml_out.write("\n")
        if not np.all(self.bridge==None):
            yaml_out.write("bridge:\n")
            for i in range(1,len(self.bridge[0])):
                parent_idx = self.bridge[0][i]
                yaml_out.write("  - [{}, {}, {}, {}]\n".format( \
                    parent_idx, np.int(self.bridge[1][parent_idx,i,1]), \
                    i, np.int(self.bridge[1][parent_idx,i,2])))
            yaml_out.write("\n")
        yaml_out.write("decomp_edge:\n")            
        for i in range(len(decomp_edge)):
            yaml_out.write("  - [{}, {}, {}, {}]\n".format( \
                decomp_edge[i][0][0], decomp_edge[i][0][1], \
                decomp_edge[i][1][0], decomp_edge[i][1][1]))
        yaml_out.close()
    
    def load_decomp(self, filedir='poly_decomp', filename='result.yaml'):
        decomp_edge = []
        yamlFile = os.path.join(filedir, filename)
        with open(yamlFile, "r") as input_stream:
            yaml_in = yaml.load(input_stream)
            decomp_raw = np.array(yaml_in["decomp_edge"])
            for i in range(decomp_raw.shape[0]):
                decomp_edge.append([[decomp_raw[i][0],decomp_raw[i][1]],[decomp_raw[i][2],decomp_raw[i][3]]])
        return decomp_edge
    
    def plot_approx(self):
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        point_x = np.zeros(len(self.points)+1)
        point_y = np.zeros(len(self.points)+1)
        for i in range(len(self.points)):
            point_x[i] = self.points[i].x
            point_y[i] = self.points[i].y
            ax.text(point_x[i], point_y[i], "{}".format(i), fontsize=15)
        point_x[-1] = self.points[0].x
        point_y[-1] = self.points[0].y
        ax.plot(point_x, point_y, '-', linewidth=1, color='black')
        plt.grid()
        plt.show()
        plt.pause(0.1)
        return

###########################################################################################
# Generate graph
#  - Generate convex polygon group
#  - Find start, end polygon
#  - generate graph
def load_decomp_data(filedir='poly_decomp', filename='result.yaml'):
    points_set = []
    original_edge = []
    decomp_edge = []
    yamlFile = os.path.join(filedir, filename)
    with open(yamlFile, "r") as input_stream:
        yaml_in = yaml.load(input_stream)
        points_raw = yaml_in["points"]
        for idx in points_raw.keys():
            points_set.append(points_raw[idx])
            for j in range(len(points_raw[idx])):
                original_edge.append([[idx,j],[idx,(j+1)%len(points_raw[idx])]])
        decomp_raw = np.array(yaml_in["decomp_edge"])
        for i in range(decomp_raw.shape[0]):
            decomp_edge.append([[decomp_raw[i][0],decomp_raw[i][1]],[decomp_raw[i][2],decomp_raw[i][3]]])
        if "bridge" in yaml_in.keys():
            bridge_raw = np.array(yaml_in["bridge"])
            for i in range(bridge_raw.shape[0]):
                decomp_edge.append([[bridge_raw[i][0],bridge_raw[i][1]],[bridge_raw[i][2],bridge_raw[i][3]]])
    return points_set, original_edge, decomp_edge


def check_visibility(points_set, original_edge, decomp_edge, edges, point_c = None):
    points_idx = [x[0] for x in edges]
    if np.any(point_c==None):
        points_center_t = np.mean(np.array([points_set[x[0]][x[1]] for x in points_idx]),axis=0)
        points_center = _Point(points_center_t[0], points_center_t[1])
    else:
        points_center = _Point(point_c[0], point_c[1])
        
    points_face = [_Point(points_set[x[0][0]][x[0][1]][0]+points_set[x[1][0]][x[1][1]][0],
                          points_set[x[0][0]][x[0][1]][1]+points_set[x[1][0]][x[1][1]][1])*0.5
                   for x in edges]
    flag_center = True
    for e in decomp_edge+original_edge:
        if e not in edges and [e[1],e[0]] not in edges:
            point_t1 = _Point(points_set[e[0][0]][e[0][1]][0],points_set[e[0][0]][e[0][1]][1])
            point_t2 = _Point(points_set[e[1][0]][e[1][1]][0],points_set[e[1][0]][e[1][1]][1])
            for p in points_face:
                if checkIntersect(point_t1, point_t2, points_center, p):
                    flag_center = False
        else:
            point_t1 = _Point(points_set[e[0][0]][e[0][1]][0],points_set[e[0][0]][e[0][1]][1])
            point_t2 = _Point(points_set[e[1][0]][e[1][1]][0],points_set[e[1][0]][e[1][1]][1])
            num_intersect = 0
            for p in points_face:
                if checkIntersect(point_t1, point_t2, points_center, p):
                    num_intersect += 1
            if num_intersect >= 2:
                flag_center = False
    return flag_center

def get_polygon_set(points_set, original_edge, decomp_edge):
    polygon_set = []
    search_history = []

    def find_polygon(start_node, prev_node=None, polygon_vertex=[]):
        connected_nodes = []
        # find connected node
        for e in decomp_edge+original_edge:
            if start_node == e[0]:
                if prev_node == None:
                    connected_nodes.append(e[1])
                else:
                    if prev_node != e[1]:
                        connected_nodes.append(e[1])
            elif start_node == e[1]:
                if prev_node == None:
                    connected_nodes.append(e[0])
                else:
                    if prev_node != e[0]:
                        connected_nodes.append(e[0])
        prev_angle = None
        if prev_node != None:
            prev_angle = np.arctan2(
                points_set[prev_node[0]][prev_node[1]][1]- points_set[start_node[0]][start_node[1]][1],
                points_set[prev_node[0]][prev_node[1]][0]- points_set[start_node[0]][start_node[1]][0])
        nodes_angle = []
        for idx, node_t in enumerate(connected_nodes):
            nodes_angle.append([np.arctan2(
                points_set[node_t[0]][node_t[1]][1]- points_set[start_node[0]][start_node[1]][1],
                points_set[node_t[0]][node_t[1]][0]- points_set[start_node[0]][start_node[1]][0]),idx])
        nodes_angle.sort(key=lambda x: x[0])

        for i in range(1,len(nodes_angle)+1):
            next_node = connected_nodes[nodes_angle[i-1][1]]
            flag_new_cycle = False
            if prev_angle == None:
                flag_new_cycle = True
            else:
                if i<len(nodes_angle) and (prev_angle >= nodes_angle[i%len(nodes_angle)][0] or prev_angle <= nodes_angle[i-1][0]):
                    flag_new_cycle = True

            if not flag_new_cycle:
                if leftOn(
                    _Point(points_set[prev_node[0]][prev_node[1]][0], points_set[prev_node[0]][prev_node[1]][1]),
                    _Point(points_set[start_node[0]][start_node[1]][0], points_set[start_node[0]][start_node[1]][1]),
                    _Point(points_set[next_node[0]][next_node[1]][0], points_set[next_node[0]][next_node[1]][1])):
                    if next_node not in polygon_vertex:
                        polygon_vertex_t = copy.deepcopy(polygon_vertex)
                        polygon_vertex_t.append(next_node)
                        search_history.append([prev_node,start_node,next_node])
                        find_polygon(start_node=copy.deepcopy(next_node), 
                                     prev_node=copy.deepcopy(start_node),
                                     polygon_vertex=copy.deepcopy(polygon_vertex_t))
                    else:
                        idx_t = polygon_vertex.index(next_node)
                        p_v_t2 = polygon_vertex[idx_t:]
                        edges_tmp = [[p_v_t2[k],p_v_t2[(k+1)%len(p_v_t2)]] for k, x in enumerate(p_v_t2)]
                        if check_visibility(points_set, original_edge, decomp_edge, edges_tmp):
                            flag_new = False
                            parent_loop_idx = p_v_t2[0][0]
                            for j in range(1,len(p_v_t2)):
                                if parent_loop_idx != p_v_t2[j][0]:
                                    flag_new = True
                            if flag_new == False and parent_loop_idx == 0:
                                flag_new = True
                            for j in range(len(p_v_t2)):
                                p_v_t3 = p_v_t2[j:]+p_v_t2[:j]
                                if p_v_t3 in polygon_set:
                                    flag_new = False
                            if flag_new:
                                polygon_set.append(p_v_t2)
#                                 print("add polygon: ", edges_tmp)
#                             else:
#                                 print("add polygon fail: ", edges_tmp)
                        else:
                            polygon_vertex = []
            else:
                prev_node_t = connected_nodes[nodes_angle[i%len(nodes_angle)][1]]
                if leftOn(
                    _Point(points_set[prev_node_t[0]][prev_node_t[1]][0], points_set[prev_node_t[0]][prev_node_t[1]][1]),
                    _Point(points_set[start_node[0]][start_node[1]][0], points_set[start_node[0]][start_node[1]][1]),
                    _Point(points_set[next_node[0]][next_node[1]][0], points_set[next_node[0]][next_node[1]][1])):
                    if [prev_node_t,start_node,next_node] not in search_history:
                        search_history.append([prev_node_t,start_node,next_node])
                        find_polygon(start_node=copy.deepcopy(next_node), 
                                     prev_node=copy.deepcopy(start_node), 
                                     polygon_vertex=copy.deepcopy([prev_node_t,start_node,next_node]))
        return

    find_polygon([0,0])
    return polygon_set

def get_polygon_path(points_set, original_edge, decomp_edge, polygon_set, initial_point, final_point):
    # Find initial and final polygon
    initial_polygon_idx = None
    final_polygon_idx = None
    for i in range(len(polygon_set)):
        edges_tmp = [[polygon_set[i][k],polygon_set[i][(k+1)%len(polygon_set[i])]] for k, x in enumerate(polygon_set[i])]
        if check_visibility(points_set, original_edge, decomp_edge, edges_tmp, initial_point):
            initial_polygon_idx = i
        if check_visibility(points_set, original_edge, decomp_edge, edges_tmp, final_point):
            final_polygon_idx = i
    print("initial_polygon_idx: ",initial_polygon_idx,", final_polygon_idx: ",final_polygon_idx)

    # Generate Graph
    face_vertex = []
    for i in range(len(polygon_set)):
        if i == initial_polygon_idx:
            points_center = _Point(initial_point[0], initial_point[1])
        elif i == final_polygon_idx:
            points_center = _Point(final_point[0], final_point[1])
        else:
            points_center_t = np.mean(np.array([points_set[x[0]][x[1]] for x in polygon_set[i]]),axis=0)
            points_center = _Point(points_center_t[0], points_center_t[1])
        face_vertex.append([0,points_center,i])
    for e in decomp_edge+original_edge:
        points_center_t = np.mean(np.array([points_set[x[0]][x[1]] for x in e]),axis=0)
        points_center = _Point(points_center_t[0], points_center_t[1])
        face_vertex.append([1,points_center,e])

    face_graph = np.zeros((len(face_vertex),len(face_vertex)))
    for i in range(len(polygon_set)):
        for idx, vtx1 in enumerate(polygon_set[i]):
            vtx2 = polygon_set[i][(idx+1)%len(polygon_set[i])]
            curr_edge_idx = None
            for edge_idx, face_vtx in enumerate(face_vertex[len(polygon_set):]):
                if [vtx1,vtx2] == face_vtx[2] or [vtx2,vtx1] == face_vtx[2]:
                    curr_edge_idx = edge_idx+len(polygon_set)
                    break
            face_graph[i,curr_edge_idx] = np.sqrt(sqdist(face_vertex[i][1],face_vertex[curr_edge_idx][1]))
            face_graph[curr_edge_idx,i] = np.sqrt(sqdist(face_vertex[i][1],face_vertex[curr_edge_idx][1]))

    # Dijstra to find polygon path
    g = Graph(len(face_vertex))
    g.graph = list(face_graph)
    dist_t, parent_t = g.dijkstra(initial_polygon_idx)
    polygon_path = [final_polygon_idx]
#     while True:
    for i in range(len(face_vertex)):
        next_idx = parent_t[polygon_path[-1]] 
        polygon_path.append(next_idx)
        if next_idx == initial_polygon_idx:
            break
    polygon_path.reverse()
    return face_vertex, face_graph, polygon_path


# Save & load result
def save_polygon_path(points_set, polygon_set, face_vertex, polygon_path, initial_point, final_point, \
                      filedir='poly_decomp', filename='result_polygon_path.yaml', sample_name='test', t_set=None):
    yamlFile = os.path.join(filedir, filename)
    yaml_out = open(yamlFile,"w")
    yaml_out.write("{}:\n".format(sample_name))
    yaml_out.write("    points:\n")
    for i in range(len(points_set)):
        yaml_out.write("        {}:\n".format(i))
        for j in range(len(points_set[i])):
            yaml_out.write("          - [{}, {}]\n".format(points_set[i][j][0],points_set[i][j][1]))
    yaml_out.write("\n")
    yaml_out.write("    polygon_set:\n")
    for i in range(len(polygon_set)):
        yaml_out.write("        {}:\n".format(i))
        for j in range(len(polygon_set[i])):
            yaml_out.write("          - [{}, {}]\n".format(polygon_set[i][j][0],polygon_set[i][j][1]))
    yaml_out.write("\n")
    yaml_out.write("    face_vertex:\n")
    for i in range(len(face_vertex)):
        if face_vertex[i][0] == 0:
            yaml_out.write("      - [{}, {}, {}, {}]\n".format( \
                face_vertex[i][0], face_vertex[i][1].x, face_vertex[i][1].y, \
                face_vertex[i][2]))
        elif face_vertex[i][0] == 1:
            yaml_out.write("      - [{}, {}, {}, {}, {}, {}, {}]\n".format( \
                face_vertex[i][0], face_vertex[i][1].x, face_vertex[i][1].y, \
                face_vertex[i][2][0][0], face_vertex[i][2][0][1], \
                face_vertex[i][2][1][0], face_vertex[i][2][1][1]))
    yaml_out.write("\n")
    yaml_out.write("    polygon_path: [{}]\n".format(', '.join([str(x) for x in polygon_path])))
    yaml_out.write("\n")
    yaml_out.write("    initial_point: [{}]\n".format(', '.join([str(x) for x in initial_point])))
    yaml_out.write("\n")
    yaml_out.write("    final_point: [{}]\n".format(', '.join([str(x) for x in final_point])))
    yaml_out.write("\n")
    if np.any(t_set == None):
        t_set = np.ones(np.int((len(polygon_path)+1)/2))
    yaml_out.write("    t_set: [{}]\n".format(', '.join([str(x) for x in t_set])))
    yaml_out.close()
    return

def load_polygon_path(filedir='poly_decomp', filename='result_polygon_path.yaml', sample_name='test', flag_t_set=False):
    points_set = []
    polygon_set = []
    face_vertex = []
    polygon_path = []
    
    yamlFile = os.path.join(filedir, filename)
    with open(yamlFile, "r") as input_stream:
        yaml_in = yaml.load(input_stream)
        yaml_in = yaml_in[sample_name]
        points_raw = yaml_in["points"]
        for idx in points_raw.keys():
            points_set.append(points_raw[idx])
        
        polygon_set_raw = yaml_in["polygon_set"]
        for idx in polygon_set_raw.keys():
            polygon_set.append(polygon_set_raw[idx])
        
        face_vertex_raw = yaml_in["face_vertex"]
        for idx in range(len(face_vertex_raw)):
            if face_vertex_raw[idx][0] == 0:
                face_vertex.append([0,
                    _Point(face_vertex_raw[idx][1],face_vertex_raw[idx][2]),
                    face_vertex_raw[idx][3]])
            elif face_vertex_raw[idx][0] == 1:
                face_vertex.append([1,
                    _Point(face_vertex_raw[idx][1],face_vertex_raw[idx][2]),
                    [[face_vertex_raw[idx][3],face_vertex_raw[idx][4]],
                    [face_vertex_raw[idx][5],face_vertex_raw[idx][6]]]])
        
        polygon_path = yaml_in["polygon_path"]
        initial_point = yaml_in["initial_point"]
        final_point = yaml_in["final_point"]
        if flag_t_set:
            t_set = np.array(yaml_in["t_set"])
    
    if flag_t_set:
        return points_set, polygon_set, face_vertex, polygon_path, initial_point, final_point, t_set
    else:
        return points_set, polygon_set, face_vertex, polygon_path, initial_point, final_point

#########################
# Plot
#########################
def plot_polygon_path(points_set, polygon_set, \
                    face_vertex, face_graph, polygon_path, \
                    initial_point, final_point, points_set_plot=None, \
                    plot_mode=0, flag_save=False, save_dir="./poly_path.pdf"):    
    if points_set_plot == None:
        points_set_plot = copy.deepcopy(points_set)
    
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    
    # Plot background
    for i in range(len(points_set_plot)):
        point_x = np.zeros(len(points_set_plot[i])+1)
        point_y = np.zeros(len(points_set_plot[i])+1)
        for j in range(len(points_set_plot[i])):
            point_x[j] = points_set_plot[i][j][0]
            point_y[j] = points_set_plot[i][j][1]
        point_x[-1] = points_set_plot[i][0][0]
        point_y[-1] = points_set_plot[i][0][1]
        if i == 0:
            ax.plot(point_x, point_y, '-', linewidth=5, color='black')
        else:
            ax.plot(point_x, point_y, '-', linewidth=5, color='black')
            ax.fill(point_x, point_y, color='black', alpha=0.2)

    # Plot initial and final point
    plt.scatter(initial_point[0], initial_point[1], s=300, c='tab:green', marker=(5, 1))
    plt.text(initial_point[0], initial_point[1]+0.3, "start", fontsize=20, ha='center')
    plt.scatter(final_point[0], final_point[1], s=300, c='tab:red', marker=(5, 1))
    plt.text(final_point[0], final_point[1]+0.3, "end", fontsize=20, ha='center')

    # Plot convex polygon decomposition
    if plot_mode >= 1:
        for _poly_idx, _poly in enumerate(polygon_set):
            point_x = np.zeros(len(_poly)+1)
            point_y = np.zeros(len(_poly)+1)
            for idx, vtx in enumerate(_poly):
                point_x[idx] = points_set[vtx[0]][vtx[1]][0]
                point_y[idx] = points_set[vtx[0]][vtx[1]][1]
            point_x[-1] = points_set[_poly[0][0]][_poly[0][1]][0]
            point_y[-1] = points_set[_poly[0][0]][_poly[0][1]][1]
            ax.plot(point_x, point_y, '-', linewidth=1, color='black')
            if plot_mode >= 3:
                if _poly_idx in polygon_path:
                    ax.fill(point_x, point_y, color='tab:orange', alpha=0.2)

    # Plot face vertex
    if plot_mode == 2 or plot_mode == 3:
        for i in range(len(face_vertex)):
#             circle1 = plt.Circle((face_vertex[i][1].x, face_vertex[i][1].y), radius=0.1, color='tab:blue')
#             ax.add_artist(circle1)
            ax.scatter(face_vertex[i][1].x, face_vertex[i][1].y, s=50, color='tab:blue', zorder=2)

        # Plot face graph
        for i in range(len(face_vertex)):
            for j in range(i+1,len(face_vertex)):
                if face_graph[i,j] > 0:
                    point_x = np.zeros(2)
                    point_y = np.zeros(2)
                    point_x[0] = face_vertex[i][1].x
                    point_y[0] = face_vertex[i][1].y
                    point_x[1] = face_vertex[j][1].x
                    point_y[1] = face_vertex[j][1].y
                    ax.plot(point_x, point_y, '-', linewidth=1, color='black')

    # Plot polygon path
    if plot_mode == 3:
        point_x = np.zeros(len(polygon_path))
        point_y = np.zeros(len(polygon_path))
        for idx, vtx_idx in enumerate(polygon_path):
            vtx = face_vertex[vtx_idx][1]
            point_x[idx] = vtx.x
            point_y[idx] = vtx.y
        ax.plot(point_x, point_y, '-', linewidth=3, color='red')

    # Plot plane on polygon path
    if plot_mode == 4:
        for idx, vtx_idx in enumerate(polygon_path):
            if face_vertex[vtx_idx][0] == 1:
                edge_t = face_vertex[vtx_idx][2]
                point_x = np.zeros(2)
                point_y = np.zeros(2)
                point_x[0] = points_set[edge_t[0][0]][edge_t[0][1]][0]
                point_y[0] = points_set[edge_t[0][0]][edge_t[0][1]][1]
                point_x[1] = points_set[edge_t[1][0]][edge_t[1][1]][0]
                point_y[1] = points_set[edge_t[1][0]][edge_t[1][1]][1]
                ax.plot(point_x, point_y, '-', linewidth=3, color='red')
        point_x = np.zeros(len(polygon_path))
        point_y = np.zeros(len(polygon_path))
        for idx, vtx_idx in enumerate(polygon_path):
            vtx = face_vertex[vtx_idx][1]
            point_x[idx] = vtx.x
            point_y[idx] = vtx.y
        ax.plot(point_x, point_y, '--', linewidth=1, color='black')

    # plt.grid()
    plt.axis('off')

    if flag_save:
        plt.savefig(save_dir)
    plt.show()
    plt.pause(0.1)
    return
    

###########################################################################################
# Connect with MFBO
#  - Generate Dataset
#  - Add polygon constraints
def get_plane_pos_set(points_set, polygon_set, face_vertex, \
    polygon_path, initial_point, final_point, 
    unit_height_t=1.):

    plane_pos_set = []
    plane_pos_set_corner_buffer = []
    polygon_path_points = []
    prev_output_edge = []
    prev_output_plane = []
    num_polygon_path = np.int((len(polygon_path)+1)/2)
    for i in range(num_polygon_path):
        if i == 0:
            polygon_path_points.append( \
                [face_vertex[polygon_path[0]][1].x, \
                face_vertex[polygon_path[0]][1].y,0])
        if i < num_polygon_path-1:
            polygon_path_points.append( \
                [face_vertex[polygon_path[2*i+1]][1].x, \
                face_vertex[polygon_path[2*i+1]][1].y,0])
        else:
            polygon_path_points.append( \
                [face_vertex[polygon_path[2*i]][1].x, \
                face_vertex[polygon_path[2*i]][1].y,0])

        polygon_idx = face_vertex[polygon_path[2*i]][2]
        if i < num_polygon_path-1:
            output_edge_t = face_vertex[polygon_path[2*i+1]][2]
            _p_idx = face_vertex[polygon_path[2*i]][2]
            edges_tmp = [[polygon_set[_p_idx][k],polygon_set[_p_idx][(k+1)%len(polygon_set[_p_idx])]] \
                         for k, x in enumerate(polygon_set[_p_idx])]
            if output_edge_t in edges_tmp:
                output_edge = output_edge_t
            elif [output_edge_t[1], output_edge_t[0]] in edges_tmp:
                output_edge = [output_edge_t[1], output_edge_t[0]]
            else:
                output_edge = None
        else:
            output_edge = None
        polygon_info_t = dict()
        polygon_info_t["input_plane"] = []
        polygon_info_t["constraints_plane"] = []
        polygon_info_t["output_plane"] = []
        polygon_info_t["corner_plane"] = []

        if len(prev_output_plane) > 0:
            polygon_info_t["input_plane"] = prev_output_plane

        if np.all(output_edge != None):    
            x_1_t = points_set[output_edge[0][0]][output_edge[0][1]][0]
            y_1_t = points_set[output_edge[0][0]][output_edge[0][1]][1]
            x_2_t = points_set[output_edge[1][0]][output_edge[1][1]][0]
            y_2_t = points_set[output_edge[1][0]][output_edge[1][1]][1]

            polygon_info_t["output_plane"].append(
                [x_2_t, y_2_t, -unit_height_t])
            polygon_info_t["output_plane"].append(
                [x_1_t, y_1_t, -unit_height_t])
            polygon_info_t["output_plane"].append(
                [x_1_t, y_1_t, unit_height_t])
            polygon_info_t["output_plane"].append(
                [x_2_t, y_2_t, unit_height_t])

        polygon_top = []
        polygon_bottom = []
        for j in range(len(polygon_set[polygon_idx])):
            j_n = (j+1)%len(polygon_set[polygon_idx])
            v_idx1 = polygon_set[polygon_idx][j]
            v_idx2 = polygon_set[polygon_idx][j_n]
            polygon_top.append(
                [points_set[v_idx1[0]][v_idx1[1]][0],
                 points_set[v_idx1[0]][v_idx1[1]][1],
                 unit_height_t])
            polygon_bottom.append(
                [points_set[v_idx1[0]][v_idx1[1]][0],
                 points_set[v_idx1[0]][v_idx1[1]][1],
                 -unit_height_t])
            if [v_idx1, v_idx2] == prev_output_edge \
                or [v_idx1, v_idx2] == output_edge:
                continue

            polygon_info_single_t = []
            polygon_info_single_t.append(
                [points_set[v_idx2[0]][v_idx2[1]][0],
                 points_set[v_idx2[0]][v_idx2[1]][1],
                 -unit_height_t])
            polygon_info_single_t.append(
                [points_set[v_idx1[0]][v_idx1[1]][0],
                 points_set[v_idx1[0]][v_idx1[1]][1],
                 -unit_height_t])
            polygon_info_single_t.append(
                [points_set[v_idx1[0]][v_idx1[1]][0],
                 points_set[v_idx1[0]][v_idx1[1]][1],
                 unit_height_t])
            polygon_info_single_t.append(
                [points_set[v_idx2[0]][v_idx2[1]][0],
                 points_set[v_idx2[0]][v_idx2[1]][1],
                 unit_height_t])
            polygon_info_t["constraints_plane"].append(polygon_info_single_t)

        polygon_top.reverse()
        polygon_info_t["constraints_plane"].append(polygon_top)
        polygon_info_t["constraints_plane"].append(polygon_bottom)


        if len(plane_pos_set_corner_buffer) > 0:
            for k in range(len(plane_pos_set_corner_buffer)):
                polygon_info_t["corner_plane"].append(plane_pos_set_corner_buffer[k])
            plane_pos_set_corner_buffer = []

        if len(polygon_info_t["input_plane"]) > 0 \
            and len(polygon_info_t["output_plane"]) > 0:
            vi0 = np.array(polygon_info_t["input_plane"][3])
            vi1 = np.array(polygon_info_t["input_plane"][2])
            vf0 = np.array(polygon_info_t["output_plane"][1])
            vf1 = np.array(polygon_info_t["output_plane"][0])

            eps_t = 1e-3
            alpha_t = 0.03
            if np.linalg.norm(vi0-vf1) < eps_t:
                polygon_info_single_t = []
                via = vi0 + (vi1-vi0)*alpha_t
                vfa = vf1 + (vf0-vf1)*alpha_t
                polygon_info_single_t.append(list(via))
                polygon_info_single_t.append(list(vfa))
                polygon_info_single_t.append([vfa[0],vfa[1],unit_height_t])
                polygon_info_single_t.append([via[0],via[1],unit_height_t])
                polygon_info_t["corner_plane"].append(polygon_info_single_t)

                _p_idx = face_vertex[polygon_path[2*(i-1)]][2]
                edges_prev_t = [[polygon_set[_p_idx][k],polygon_set[_p_idx][(k+1)%len(polygon_set[_p_idx])]] \
                         for k, x in enumerate(polygon_set[_p_idx])]
                _e_prev_idx = edges_prev_t.index([prev_output_edge[1],prev_output_edge[0]])
                _e_prev_idx = (_e_prev_idx+1)%len(polygon_set[_p_idx])
                _e_prev_idx2 = (_e_prev_idx+1)%len(polygon_set[_p_idx])
                via2 = np.array([points_set[polygon_set[_p_idx][_e_prev_idx2][0]][polygon_set[_p_idx][_e_prev_idx2][1]][0],\
                                 points_set[polygon_set[_p_idx][_e_prev_idx2][0]][polygon_set[_p_idx][_e_prev_idx2][1]][1],\
                                 -unit_height_t])
                polygon_info_single_t = []
                polygon_info_single_t.append(list(via2))
                polygon_info_single_t.append(list(via))
                polygon_info_single_t.append([via[0],via[1],unit_height_t])
                polygon_info_single_t.append([via2[0],via2[1],unit_height_t])
                plane_pos_set[-1]["corner_plane"].append(polygon_info_single_t)

                _p_idx = face_vertex[polygon_path[2*(i+1)]][2]
                edges_next_t = [[polygon_set[_p_idx][k],polygon_set[_p_idx][(k+1)%len(polygon_set[_p_idx])]] \
                         for k, x in enumerate(polygon_set[_p_idx])]
                _e_next_idx = edges_next_t.index([output_edge[1],output_edge[0]])
                _e_next_idx2 = (_e_next_idx-1)%len(polygon_set[_p_idx])
                vfa2 = np.array([points_set[polygon_set[_p_idx][_e_next_idx2][0]][polygon_set[_p_idx][_e_next_idx2][1]][0],\
                                 points_set[polygon_set[_p_idx][_e_next_idx2][0]][polygon_set[_p_idx][_e_next_idx2][1]][1],\
                                 -unit_height_t])
                polygon_info_single_t = []
                polygon_info_single_t.append(list(vfa))
                polygon_info_single_t.append(list(vfa2))
                polygon_info_single_t.append([vfa2[0],vfa2[1],unit_height_t])
                polygon_info_single_t.append([vfa[0],vfa[1],unit_height_t])
                plane_pos_set_corner_buffer.append(polygon_info_single_t)

            elif np.linalg.norm(vi1-vf0) < eps_t:
                polygon_info_single_t = []
                via = vi1 + (vi0-vi1)*alpha_t
                vfa = vf0 + (vf1-vf0)*alpha_t
                polygon_info_single_t.append(list(vfa))
                polygon_info_single_t.append(list(via))
                polygon_info_single_t.append([via[0],via[1],unit_height_t])
                polygon_info_single_t.append([vfa[0],vfa[1],unit_height_t])
                polygon_info_t["corner_plane"].append(polygon_info_single_t)

                _p_idx = face_vertex[polygon_path[2*(i-1)]][2]
                edges_prev_t = [[polygon_set[_p_idx][k],polygon_set[_p_idx][(k+1)%len(polygon_set[_p_idx])]] \
                         for k, x in enumerate(polygon_set[_p_idx])]
                _e_prev_idx = edges_prev_t.index([prev_output_edge[1],prev_output_edge[0]])
                _e_prev_idx2 = (_e_prev_idx-1)%len(polygon_set[_p_idx])
                via2 = np.array([points_set[polygon_set[_p_idx][_e_prev_idx2][0]][polygon_set[_p_idx][_e_prev_idx2][1]][0],\
                                 points_set[polygon_set[_p_idx][_e_prev_idx2][0]][polygon_set[_p_idx][_e_prev_idx2][1]][1],\
                                 -unit_height_t])
                polygon_info_single_t = []
                polygon_info_single_t.append(list(via))
                polygon_info_single_t.append(list(via2))
                polygon_info_single_t.append([via2[0],via2[1],unit_height_t])
                polygon_info_single_t.append([via[0],via[1],unit_height_t])
                plane_pos_set[-1]["corner_plane"].append(polygon_info_single_t)

                _p_idx = face_vertex[polygon_path[2*(i+1)]][2]
                edges_next_t = [[polygon_set[_p_idx][k],polygon_set[_p_idx][(k+1)%len(polygon_set[_p_idx])]] \
                         for k, x in enumerate(polygon_set[_p_idx])]
                _e_next_idx = edges_next_t.index([output_edge[1],output_edge[0]])
                _e_next_idx = (_e_next_idx+1)%len(polygon_set[_p_idx])
                _e_next_idx2 = (_e_next_idx+1)%len(polygon_set[_p_idx])
                vfa2 = np.array([points_set[polygon_set[_p_idx][_e_next_idx2][0]][polygon_set[_p_idx][_e_next_idx2][1]][0],\
                                 points_set[polygon_set[_p_idx][_e_next_idx2][0]][polygon_set[_p_idx][_e_next_idx2][1]][1],\
                                 -unit_height_t])
                polygon_info_single_t = []
                polygon_info_single_t.append(list(vfa))
                polygon_info_single_t.append([vfa[0],vfa[1],unit_height_t])
                polygon_info_single_t.append([vfa2[0],vfa2[1],unit_height_t])
                polygon_info_single_t.append(list(vfa2))
                plane_pos_set_corner_buffer.append(polygon_info_single_t)

        plane_pos_set.append(polygon_info_t)
        prev_output_edge = copy.deepcopy(output_edge)
        if np.all(prev_output_edge != None):
            prev_output_edge.reverse()
        prev_output_plane = copy.deepcopy(polygon_info_t["output_plane"])
        if np.all(prev_output_plane != None):   
            prev_output_plane.reverse()
    return plane_pos_set, polygon_path_points


#########################
# Plot
#########################
import plotly.graph_objects as go
def plot_plane_pos_set( \
    points_set, polygon_set, face_vertex, \
    polygon_path, initial_point, final_point, \
    plane_pos_set, polygon_path_points):
    mesh_data = []
    for i in range(len(plane_pos_set)):
        for j in range(len(plane_pos_set[i]["constraints_plane"])):
            p_t = np.array(plane_pos_set[i]["constraints_plane"][j])
            ijk_t = np.array([[0,k+1,k+2] for k in range(p_t.shape[0]-2)])
            mesh_t = go.Mesh3d(
                x=list(p_t[:,0]), y=list(p_t[:,1]), z=list(p_t[:,2]),
                color='lightpink', opacity=0.20, showscale=True,
                i=ijk_t[:,0], j=ijk_t[:,1], k=ijk_t[:,2],)
            mesh_data.append(mesh_t)
            c_t = np.mean(p_t,axis=0)
            v0 = np.array(plane_pos_set[i]["constraints_plane"][j][0])
            v1 = np.array(plane_pos_set[i]["constraints_plane"][j][1])
            for k in range(2,len(plane_pos_set[i]["constraints_plane"][j])):
                v2 = np.array(plane_pos_set[i]["constraints_plane"][j][k])
                res_t = np.fabs((v1-v0).dot(v2-v0)/np.linalg.norm(v1-v0)/np.linalg.norm(v2-v0))
                if res_t < 1-1e-4:
                    break
            V_norm = np.cross(v1-v0, v2-v0)*1.0
            V_norm /= np.linalg.norm(V_norm)
            c_t2 = c_t + V_norm * 0.5
            vector_t = go.Scatter3d(
                x=[c_t[0],c_t2[0]], y=[c_t[1],c_t2[1]], z=[c_t[2],c_t2[2]],
                marker = dict(size=1, color="black"),
                line = dict(color="black", width = 6))
            mesh_data.append(vector_t)
        if len(plane_pos_set[i]["input_plane"]) > 0:
            p_t = np.array(plane_pos_set[i]["input_plane"])
            ijk_t = np.array([[0,k+1,k+2] for k in range(p_t.shape[0]-2)])
            mesh_t = go.Mesh3d(
                x=list(p_t[:,0]), y=list(p_t[:,1]), z=list(p_t[:,2]),
                color='green', opacity=0.20, showscale=True,
                i=ijk_t[:,0], j=ijk_t[:,1], k=ijk_t[:,2],)
            mesh_data.append(mesh_t)
            c_t = np.mean(p_t,axis=0)
            v0 = np.array(plane_pos_set[i]["input_plane"][0])
            v1 = np.array(plane_pos_set[i]["input_plane"][1])
            for k in range(2,len(plane_pos_set[i]["input_plane"])):
                v2 = np.array(plane_pos_set[i]["input_plane"][k])
                res_t = np.fabs((v1-v0).dot(v2-v0)/np.linalg.norm(v1-v0)/np.linalg.norm(v2-v0))
                if res_t < 1-1e-4:
                    break
            V_norm = np.cross(v1-v0, v2-v0)*1.0
            V_norm /= np.linalg.norm(V_norm)
            c_t2 = c_t + V_norm * 0.5
            vector_t = go.Scatter3d(
                x=[c_t[0],c_t2[0]], y=[c_t[1],c_t2[1]], z=[c_t[2],c_t2[2]],
                marker = dict(size=1, color="green"),
                line = dict(color="green", width = 6))
            mesh_data.append(vector_t)
        if len(plane_pos_set[i]["output_plane"]) > 0:
            p_t = np.array(plane_pos_set[i]["output_plane"])
            ijk_t = np.array([[0,k+1,k+2] for k in range(p_t.shape[0]-2)])
            mesh_t = go.Mesh3d(
                x=list(p_t[:,0]), y=list(p_t[:,1]), z=list(p_t[:,2]),
                color='blue', opacity=0.20, showscale=True,
                i=ijk_t[:,0], j=ijk_t[:,1], k=ijk_t[:,2],)
            mesh_data.append(mesh_t)
            c_t = np.mean(p_t,axis=0)
            v0 = np.array(plane_pos_set[i]["output_plane"][0])
            v1 = np.array(plane_pos_set[i]["output_plane"][1])
            for k in range(2,len(plane_pos_set[i]["output_plane"])):
                v2 = np.array(plane_pos_set[i]["output_plane"][k])
                res_t = np.fabs((v1-v0).dot(v2-v0)/np.linalg.norm(v1-v0)/np.linalg.norm(v2-v0))
                if res_t < 1-1e-4:
                    break
            V_norm = np.cross(v1-v0, v2-v0)*1.0
            V_norm /= np.linalg.norm(V_norm)
            c_t2 = c_t + V_norm * 0.5
            vector_t = go.Scatter3d(
                x=[c_t[0],c_t2[0]], y=[c_t[1],c_t2[1]], z=[c_t[2],c_t2[2]],
                marker = dict(size=1, color="blue"),
                line = dict(color="blue", width = 6))
            mesh_data.append(vector_t)
        for j in range(len(plane_pos_set[i]["corner_plane"])):
            p_t = np.array(plane_pos_set[i]["corner_plane"][j])
            ijk_t = np.array([[0,k+1,k+2] for k in range(p_t.shape[0]-2)])
            mesh_t = go.Mesh3d(
                x=list(p_t[:,0]), y=list(p_t[:,1]), z=list(p_t[:,2]),
                color='orange', opacity=0.40, showscale=True,
                i=ijk_t[:,0], j=ijk_t[:,1], k=ijk_t[:,2],)
            mesh_data.append(mesh_t)
            c_t = np.mean(p_t,axis=0)
            v0 = np.array(plane_pos_set[i]["corner_plane"][j][0])
            v1 = np.array(plane_pos_set[i]["corner_plane"][j][1])
            for k in range(2,len(plane_pos_set[i]["corner_plane"][j])):
                v2 = np.array(plane_pos_set[i]["corner_plane"][j][k])
                res_t = np.fabs((v1-v0).dot(v2-v0)/np.linalg.norm(v1-v0)/np.linalg.norm(v2-v0))
                if res_t < 1-1e-4:
                    break
            V_norm = np.cross(v1-v0, v2-v0)
            V_norm /= np.linalg.norm(V_norm)
            c_t2 = c_t + V_norm * 0.5
            vector_t = go.Scatter3d(
                x=[c_t[0],c_t2[0]], y=[c_t[1],c_t2[1]], z=[c_t[2],c_t2[2]],
                marker = dict(size=1, color="red"),
                line = dict(color="red", width = 6))
            mesh_data.append(vector_t)
    fig = go.Figure(data=mesh_data)
    fig.update_layout(scene_aspectmode='data', showlegend=False)
    fig.show()
    return