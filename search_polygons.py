import copy
from shapely.geometry import LineString, Point

def start_search_edge(edge_container):
    edge_ind = edge_container[0]
    start_edge = edge_container[1]
    plane_edges_list = edge_container[2]
    xys_list = edge_container[3]
    intersection_dict_list = edge_container[4]
    current_plane_index = edge_container[5]

    start_vertex = start_edge[0]
    current_vertex = start_edge[1]
    used_vertices = [current_vertex]
    used_lines_coord = [(xys_list[start_edge[0]][0], xys_list[start_edge[0]][1]),
                        (xys_list[start_edge[1]][0], xys_list[start_edge[1]][1])]

    plane_counter_usage = {}
    for plane_index in intersection_dict_list[start_vertex]['inter_planes']:
        plane_counter_usage[plane_index] = 1
    for plane_index in intersection_dict_list[current_vertex]['inter_planes']:
        if plane_index in plane_counter_usage:
            plane_counter_usage[plane_index] += 1
        else:
            plane_counter_usage[plane_index] = 1
    plane_counter_usage[current_plane_index] = -1e6
    # print(plane_counter_usage)
    # input("Her...")

    # rest_edges_list = plane_edges_list[:edge_ind] + plane_edges_list[edge_ind + 1:]
    rest_edges_list = plane_edges_list[edge_ind + 1:]
    start_vertex_polygons_list = recursive_polygon_search(start_vertex, current_vertex,
                                                          used_vertices, used_lines_coord, rest_edges_list, xys_list, intersection_dict_list, plane_counter_usage)
    if len(start_vertex_polygons_list) == 0:
        return None
    else:
        return start_vertex_polygons_list

def recursive_polygon_search(sv, cv, uvs, ulsc, rel, xys_list, intersection_dict_list, plane_counter_usage):
    sv_polygons_list = []
    if cv == sv:
        return [[sv] + uvs]
    traversed_e_ind = []
    for e_ind, e in enumerate(rel):
        if cv in e:
            traversed_e_ind.append(e_ind)
            new_rel = [rel_el for i, rel_el in enumerate(rel) if i not in traversed_e_ind]

            # new_rel = rel
            # new_rel = rel[:e_ind] + rel[e_ind + 1:]

            cv_e_index = e.index(cv)
            new_cv = e[1 - cv_e_index]

            if new_cv in uvs:
                continue

            new_plane_counter_usage = copy.copy(plane_counter_usage)
            if new_cv != sv:
                plane_usage_overf = False
                for plane_index in intersection_dict_list[new_cv]['inter_planes']:
                    if plane_index in new_plane_counter_usage:
                        new_plane_counter_usage[plane_index] += 1
                        plane_usage_overf = plane_usage_overf or new_plane_counter_usage[plane_index] > 2
                    else:
                        new_plane_counter_usage[plane_index] = 1

                # print(uvs, new_cv)
                # print(plane_counter_usage)
                if plane_usage_overf:
                    continue

            new_uvs = uvs + [new_cv]

            if len(new_uvs) >= 2:
                line1 = LineString(ulsc[:-1]) if len(new_uvs) > 2 else Point(ulsc[0][0], ulsc[0][1])
                line2 = [(xys_list[e[0]][0], xys_list[e[0]][1]),
                         (xys_list[e[1]][0], xys_list[e[1]][1])]
                line2 = LineString(line2)

                min_dist = 1
                if line1.distance(line2) < min_dist and new_cv != sv or \
                        (len(new_uvs) > 3 and LineString(ulsc[1:-1]).distance(line2) < min_dist) or \
                                LineString([ulsc[1], ulsc[-1]]).distance(
                                    Point(ulsc[0][0], ulsc[0][1])) < min_dist:
                    continue
                line3 = LineString(ulsc[:-1] +
                                   [(xys_list[e[1 - cv_e_index]][0], xys_list[e[1 - cv_e_index]][1])])
                if line3.distance(Point(xys_list[e[cv_e_index]][0],
                                        xys_list[e[cv_e_index]][1])) < min_dist or \
                                        line1.distance(Point(xys_list[e[1 - cv_e_index]][0],
                                                             xys_list[e[1 - cv_e_index]][
                                                                 1])) < min_dist and new_cv != sv:
                    continue

            new_ulsc = ulsc + [(xys_list[e[1 - cv_e_index]][0], xys_list[e[1 - cv_e_index]][1])]

            new_ulsc_LS = LineString(new_ulsc)


            sv_polygons_list += recursive_polygon_search(sv, new_cv, new_uvs, new_ulsc, new_rel, xys_list, intersection_dict_list, new_plane_counter_usage)
    return sv_polygons_list