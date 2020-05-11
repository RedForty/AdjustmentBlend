# Imports
from maya import cmds
from functools import wraps

# NOTES --------------------------------------------------------------------- #
'''
Does not actually do position snapping of controls, so if a control is being 
space-switched, it will get destroyed during the blend.

'''
# GLOBALS ------------------------------------------------------------------- #

AXIS = ['X', 'Y', 'Z']
locator_value_graph = 'locator_value_graph'

# Helper Functions ---------------------------------------------------------- #

def map_from_to(x,a,b,c,d):
   y=(x-a)/(b-a)*(d-c)+c
   return y   

def get_curve_times(curve):
    times = cmds.keyframe(curve, q=True, timeChange=True)
    int_range = [x * 1.0 for x in xrange(int(times[0]), int(times[-1]))]
    set_range = set(int_range)
    set_range.update(times)
    list_range = list(set_range)
    list_range.sort()
    return list_range
    
def get_value_graph(curve):
    times = get_curve_times(curve)
    values = []
    for time in times:
        value = cmds.keyframe(curve, q=True, eval=True, time=(time,), valueChange=True)[0]
        values.append(value)

    value_graph = []
    for i in xrange(len(values)):
        if i > 0:
            current_value = values[i]
            previous_value = values[i-1]
            value_graph.append(abs(current_value - previous_value))
        else:
            value_graph.append(0.0)
    
    # If it is a flat curve, return it unnormalized
    value_graph = normalize_value_graph(value_graph)

    return value_graph

def normalize_value_graph(graph):
    if abs(sum(graph)) > 0.0:
        # Normalize it to 100
        mult =  100 / abs(sum(graph))
        graph = [(x * mult) for x in graph]
    return graph

# Wrapper ------------------------------------------------------------------- #
def undo(func):
    '''
    Decorator - open/close undo chunk
    '''
    @wraps(func)
    def wrap(*args, **kwargs):
        cmds.undoInfo(openChunk = True)
        try:
            return func(*args, **kwargs)
        except Exception:
            raise # will raise original error
        finally:
            cmds.undoInfo(closeChunk = True)
            # cmds.undo()

    return wrap


# The main show ------------------------------------------------------------- #
@undo
def adjustment_blend_selected(debug=False):

    baseAnimationLayer = cmds.animLayer(query=True, root=True)
    sel = cmds.ls(sl=1)
    
    if not sel:
        cmds.warning("Must select something first")
        return

    # This is just to visualize
    visualizer = None
    if debug == True:
        if cmds.objExists(locator_value_graph):
            cmds.delete(locator_value_graph)
        visualizer_shape = cmds.createNode('locator', ss=True)
        visualizer_transform = cmds.listRelatives(visualizer_shape, parent=True)[0]
        visualizer = cmds.rename(visualizer_transform, locator_value_graph)

    for control in sel:
        affected_layers = cmds.animLayer(query=True, affectedLayers=True)

        if affected_layers == None:
            cmds.warning("No animLayers for selected object")
            return

        offset_layer = ''
        driven_curves = {}
        if len(affected_layers) == 2:
            for layer in affected_layers:
                if layer != baseAnimationLayer:
                    offset_layer = layer
                    animated_attributes = cmds.listAnimatable()
                    for attribute in animated_attributes: 
                        driven_curve = cmds.animLayer(layer, q=True, findCurveForPlug=attribute)
                        original_curve = cmds.animLayer(baseAnimationLayer, q=True, findCurveForPlug=attribute)

                        if driven_curve: # If this channel is in the animLayer...
                            if cmds.keyframe(original_curve[0], q=True, keyframeCount=True) > 1:
                                if cmds.keyframe(driven_curve[0], q=True, keyframeCount=True) > 1:
                                    value_graph = get_value_graph(original_curve[0])
                                    if abs(sum(value_graph)) > 0.0:
                                        driven_curves[attribute] = [driven_curve[0], original_curve[0], value_graph]
                                    else:
                                        # Be 'smart' about rotates and translates by bundling them for stragglers
                                        # Check neighboring axis 
                                        other_value_graphs = {} 
                                        current_axis = [x for x in AXIS if(x in attribute.split('.')[-1])] or None
                                        if current_axis:
                                            for axis in AXIS: # Sorry
                                                if axis != current_axis[0]:
                                                    adjacent_attribute = attribute.split('.')[-1].replace(current_axis[0], axis)
                                                    new_adjacent_attribute_name = attribute.replace(attribute.split('.')[-1], adjacent_attribute)
                                                    adjacent_curve = cmds.animLayer(baseAnimationLayer, q=True, findCurveForPlug=new_adjacent_attribute_name)
                                                    value_graph = get_value_graph(adjacent_curve)
                                                    other_value_graphs[axis] = value_graph
                                            # Add the other axis together to see if it's non-zero
                                            added_graphs = [0.0 for x in xrange(len(value_graph))]
                                            for k, v in other_value_graphs.items():
                                                for i in xrange(len(v)):
                                                    added_graphs[i] += v[i]
                                            # Normalize the added graphs
                                            normalized_added_graphs = normalize_value_graph(added_graphs)
                                            driven_curves[attribute] = [driven_curve[0], original_curve[0], normalized_added_graphs]


            for attribute, data in driven_curves.items():
                times = get_curve_times(data[1])
                
                # This is just to visualize
                if visualizer:
                    for i in xrange(len(times)):
                        cmds.setKeyframe(visualizer + '.' + attribute.split('.')[-1], time=(times[i],), value=normalized_value_curve[i])

                layer_values = []
                for i in xrange(len(times)):    
                    layer_values.append(cmds.keyframe(data[0], q=True, eval=True, time=(times[i],), valueChange=True, absolute=True)[0])

                new_value_curve = []
                sum_percentage = 0.0
                for i in xrange(len(data[2])):
                    sum_percentage += data[2][i]
                    new_value = map_from_to(sum_percentage, 0, 100, layer_values[0], layer_values[-1])
                    new_value_curve.append(new_value)

                for i in times:
                    index = times.index(i)
                    cmds.setKeyframe(data[0], animLayer=offset_layer, time=(i,), value=new_value_curve[index])
                    # cmds.keyframe(driven_curve, e=True, time=(times[i],), valueChange=new_value_curve[i])
        else:
            cmds.warning('Too many animation layers. Only takes two, "BaseAnimation" and one extra animLayer.')
            return


# adjustment_blend_selected()
