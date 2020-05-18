# Imports
from maya import cmds
from functools import wraps

# Notes:
'''    
To use:
1. Have an animated object
2. Add object to only ONE animation layer. We'll call this the adjustment layer.
3. Add an adjustment to the pose on that layer on the first frame and last frame of the animation.
4. Select the object
5. Then run: adjustment_blend_selected()

'''
# GLOBALS ------------------------------------------------------------------- #

AXIS = ['X', 'Y', 'Z']


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
    
    
# Helper Functions ---------------------------------------------------------- #

def map_from_to(x,a,b,c,d):
   y=(x-a)/(b-a)*(d-c)+c
   return y   

def get_layer_membership(attribute):
    '''
    This function returns a list of animationLayers the attribute belongs to.
    '''
    layer_membership = []
    root_layer = cmds.animLayer(query=True, root=True)
    layer_membership.append(root_layer)
    child_layers = cmds.animLayer(root_layer, q=True, children=True) or []
    
    for layer in child_layers:
        if attribute in cmds.animLayer(layer, q=True, attribute=True):
            layer_membership.append(layer)
    return layer_membership

def get_curve_range(curve, measurement_curve=None):
    # Gets range plus any keys that are not on whole frames - floats
    times = cmds.keyframe(curve, q=True, timeChange=True)
    crop_range = []
    if measurement_curve:
        crop_range = cmds.keyframe(measurement_curve, q=True, timeChange=True)
    times.extend(crop_range)
    int_range = [x * 1.0 for x in xrange(int(min(times)), int(max(times))+1)]
    set_range = set(int_range)
    set_range.update(times)
    list_range = list(set_range)
    list_range.sort()
    
    if measurement_curve:
        list_range = [x for x in list_range if x >= min(crop_range) and x <= max(crop_range)]

    return list_range

def normalize_values(values, normal=100):
    # Normalize it to 100
    if abs(sum(values)) > 0.0:
        mult =  normal / abs(sum(values))
        return [(x * mult) for x in values]
    else:
        # cmds.error('Curve cannot be normalized. Values are flat.')
        return None


# Go time here -------------------------------------------------------------- #

@undo
def adjustment_blend_selected():

    sel = cmds.ls(sl=1)
    if not sel:
        cmds.warning("Nothing happened. Nothing selected.")
        return None
    
    for thing in sel:

        cmds.select(clear=True)
        cmds.select(thing)
        attributes = cmds.listAnimatable()

        for attribute in attributes:
            attribute = attribute.split('|')[-1]
            layer_membership = get_layer_membership(attribute)
            if len(layer_membership) > 2:
                cmds.error("Selected channel belongs to more than two layers -> {}".format(layer_membership))
                cmds.warning("For proper use, the channel should belong to only BaseAnimation plus one anim layer to composite the offset.")
                continue
            if len(layer_membership) < 2:
                print "No adjustment curve found for {}.".format(attribute)
                continue

            # This is much simpler:
            layers_curves = {}
            for layer in layer_membership:
                curves = cmds.animLayer(layer, q=True, findCurveForPlug=attribute) or []
                for curve in curves:
                    layers_curves[layer] = curve

            adjustment_range = []
            adjustment_curve = ''
            base_curve = ''
            adjustment_layer = ''
            base_layer = ''
            for layer, curve in layers_curves.items():
                if layer != cmds.animLayer(query=True, root=True):
                    times = cmds.keyframe(curve, q=True, timeChange=True)
                    adjustment_range = [min(times), max(times)]
                    adjustment_curve = curve
                    adjustment_layer = layer
                else:
                    base_curve = curve
                    base_layer = layer

            adjustment_range_filled = get_curve_range(base_curve, adjustment_curve)
            if len(adjustment_range_filled) < 3:
                print "No meaningful adjustment found for {0}".format(attribute)
                continue
            
            base_curve_values = []
            for time in adjustment_range_filled:
                base_curve_values.append(cmds.keyframe(base_curve, query=True, time=(time,), eval=True, valueChange=True, absolute=True)[0])

            base_value_graph = [0.0]
            for i in xrange(len(base_curve_values)):
                if i > 0:
                    current_value = base_curve_values[i]
                    previous_value = base_curve_values[i-1]
                    base_value_graph.append(abs(current_value - previous_value))
            
            adjustment_curve_values = [] 
            for time in adjustment_range_filled: 
                adjustment_curve_values.append(cmds.keyframe(adjustment_curve, query=True, time=(time,), eval=True, valueChange=True, absolute=True)[0])

            adjustment_value_graph = [0.0]
            for i in xrange(len(adjustment_curve_values)):
                if i > 0:
                    current_value = adjustment_curve_values[i]
                    previous_value = adjustment_curve_values[i-1]
                    adjustment_value_graph.append(abs(current_value - previous_value))

            if sum(adjustment_value_graph) > sum(base_value_graph):
                print "sum({0}) is larger than sum({1}). This *may* lead to weird results. Or it might be fine.".format(adjustment_curve, base_curve)

            normalized_adjustment_value_graph = normalize_values(adjustment_value_graph)

            if abs(sum(base_value_graph)) > 0.0:
                normalized_base_value_graph = normalize_values(base_value_graph)
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
                            
                            adjacent_curve_values = []
                            for time in adjustment_range_filled:
                                adjacent_curve_values.append(cmds.keyframe(adjacent_curve, query=True, time=(time,), eval=True, valueChange=True, absolute=True)[0])
                            
                            adjacent_curve_value_graph = [0.0]
                            for i in xrange(len(adjacent_curve_values)):
                                if i > 0:
                                    current_value = adjacent_curve_values[i]
                                    previous_value = adjacent_curve_values[i-1]
                                    adjacent_curve_value_graph.append(abs(current_value - previous_value))

                            other_value_graphs[axis] = adjacent_curve_value_graph
                    # Add the other axis together to see if it's non-zero

                    added_graphs = [0.0 for x in adjustment_range_filled]
                    for k, values in other_value_graphs.items():
                        for i, value in enumerate(values):
                            added_graphs[i] += value
                    # Normalize the added graphs
                    normalized_base_value_graph = normalize_value_graph(added_graphs)

            if normalized_base_value_graph:
                adjustment_curve_keys = cmds.keyframe(adjustment_curve, q=True, timeChange=True)
                adjustment_spans = zip(adjustment_curve_keys,adjustment_curve_keys[1:])
                
                # Ok, this area got FUGLY but it works. Sorry. 
                # TODO: Come back and make this not look like shit.
                
                for span in adjustment_spans:
                    if span[-1] - 1.0 == span[0]:
                        continue # Ignore please
                    new_value_curve = []
                    sum_percentage = 0.0
                    
                    adjustment_range_filled_cropped = []
                    adjustment_curve_values_cropped = []
                    cropped_base_value_graph = []
                    for index, time in enumerate(adjustment_range_filled):
                        if time >= min(span) and time <= max(span):
                            adjustment_range_filled_cropped.append(time)
                            adjustment_curve_values_cropped.append(adjustment_curve_values[index])
                            cropped_base_value_graph.append(normalized_base_value_graph[index])
                    normalized_cropped_base_value_graph = normalize_values(cropped_base_value_graph)
                    if normalized_cropped_base_value_graph:
                        for i in xrange(len(adjustment_range_filled_cropped)):
                            
                            sum_percentage += normalized_cropped_base_value_graph[i] # This will add up to 100 along the life of the curve
                            new_value = map_from_to(sum_percentage, 0, 100, adjustment_curve_values_cropped[0], adjustment_curve_values_cropped[-1])
                            new_value_curve.append(new_value)
                        # Now set the keys
                        for index, time in enumerate(adjustment_range_filled_cropped):
                            cmds.setKeyframe(adjustment_curve, animLayer=adjustment_layer, time=(time,), value=new_value_curve[index])
                    else:
                        for index, time in enumerate(adjustment_range_filled):
                            cmds.setKeyframe(adjustment_curve, animLayer=adjustment_layer, time=(time,), value=adjustment_curve_values[index])
            else:
                for index, time in enumerate(adjustment_range_filled):
                    cmds.setKeyframe(adjustment_curve, animLayer=adjustment_layer, time=(time,), value=adjustment_curve_values[index])
    

    cmds.select(clear=True)
    cmds.select(sel)

