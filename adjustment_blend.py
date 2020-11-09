# Get selected animlayers
# Top one is the adjustment layer
# sum the curves from the lower layers

# if no selection, the top layer is the adjustment layer
# it will sum all layers below it

# It is assuming all layers involved with the selection are additive
# TODO: properly validate/calculate curves that are override

from maya import cmds, mel
import maya.api.OpenMaya as om
import maya.api.OpenMayaAnim as oma

DO_SET = True
DEBUG = False
ATTRIBUTES = [ 'translateX'
             , 'translateY'
             , 'translateZ'
             , 'rotateX'
             , 'rotateY'
             , 'rotateZ'
             , 'scaleX'
             , 'scaleY'
             , 'scaleZ'
             ]

# Helper dict will create a new key if it doesn't already exist
class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value


def _get_anim_layers():
    layer_stack = []
    root_layer = cmds.animLayer(query=True, root=True)
    if root_layer:
        layer_stack.append(root_layer)
        layer_stack.extend(cmds.animLayer(root_layer, q=True, children=True) or []) # top-down is last-to-first
    return layer_stack


def get_layers_to_process():
    # Get all layers in order in the stack
    all_layers = _get_anim_layers()

    # Get selected animLayers
    selected_layers = cmds.treeView("AnimLayerTabanimLayerEditor", q=True, selectItem=True) or []
    
    layers_to_process = all_layers[:] # Copy the list
    if len(selected_layers) == 0 or selected_layers[-1] == 'BaseAnimation': # Why select base?? Treat it like selecting nothing, I guess.
        selected_layers = [layers_to_process.pop()] # Make it a list
    elif len(selected_layers) == 1:
        index = layers_to_process.index(selected_layers[-1])
        del layers_to_process[index:]
    elif len(selected_layers) > 1:
        layers_to_process = selected_layers[:-1]

    if DEBUG:
        print 'adjustment layer target is {0}. Summed layers are {1}'.format(selected_layers[-1], layers_to_process)
    
    adjustment_layer = selected_layers[-1]
    
    if not isinstance(layers_to_process, list): layers_to_process = [layers_to_process]
    # if not isinstance(adjustment_layer, list): adjustment_layer = [adjustment_layer]
    return adjustment_layer, layers_to_process


def get_curves_to_process(obj, adjustment_layer, layers_to_process):
    ''' Returns a dictionary with the object, attribute, layer memberships, and associated curves 
        Also sanitizes the output for proper processing. This might be a point of refactor later on.
    '''
    # sel = cmds.ls(sl=1)
    # cmds.select(obj, replace=True)
    # attributes = cmds.listAnimatable() # There is probably an API version of this
    attributes = get_animated_attributes(obj) # There is probably an API version of this
    clean_attributes = []
    for attribute in attributes:
        if any(x in attribute for x in ATTRIBUTES): # Filter out non ATTRIBUTES
           clean_attributes.append(attribute)
    clean_attributes = [x.split('|')[-1] for x in clean_attributes]

    layer_attributes = Vividict()
    root_layer = cmds.animLayer(query=True, root=True)
    for layer in layers_to_process + [adjustment_layer]:
        if layer == root_layer: 
            layer_attributes[root_layer] = clean_attributes
        else:
            temp_attributes = cmds.animLayer(layer, q=True, attribute=True) or []
            layer_attributes[layer] = [x for x in temp_attributes if x in clean_attributes]
        
    if len(layer_attributes) == 1 and layer_attributes.keys() == ['BaseAnimation']:
        cmds.error("Could not find a layer membership for {}".format(obj))
        return None
    
    attribute_curves = Vividict()
    # attribute = 'pCube2.rotateY' # Helper line for debug purposes
    for attribute in clean_attributes:
        attribute_curve = Vividict()
        layer_membership = ['BaseAnimation'] # Everything is always a member of base.
        for layer, members in layer_attributes.items():
            for member in members:
                if attribute in member:
                    anim_curve = cmds.animLayer(layer, q=True, findCurveForPlug=attribute) or []
                    # if isinstance(anim_curve, list): anim_curve = anim_curve[0]
                    attribute_curve[attribute][layer] = anim_curve
                    layer_membership.append(layer)
        if layer_membership == ['BaseAnimation']: # If it's just base, it's not enough.
            continue
            
        # Make sure its a member of the adjustment layer.
        if adjustment_layer not in layer_membership: 
            continue
        else: # Success!
            # Get the list ready for the next check.
            layer_membership.remove(adjustment_layer) 
        # It must be a member of at least one of the layers to process.
        if not any(x in layer_membership for x in layers_to_process): continue
        # Make sure its not locked.
        if cmds.getAttr(attribute, lock=True): continue 
        
        # This attribute is cleared for work. Add it to the list.
        attribute_curves.update(attribute_curve)
        
    # cmds.select(sel, replace=True)
    return attribute_curves


def return_MFnAnimCurve(curve):
    msel = om.MSelectionList()
    msel.add(curve)
    mdep = msel.getDependNode(0)
    mcurve = oma.MFnAnimCurve(mdep)
    if not mcurve.name() == curve:
        return None
    return mcurve


def get_value_graph(mcurve, frange=None):
    if not frange:
        frange = get_curve_range(mcurve)
    
    values = []
    for frame in frange:
        # Get value at mtime, also feed it the current time uiUnit
        value = mcurve.evaluate(om.MTime(frame, om.MTime.uiUnit()))
        
        # Rotation curves are returned as Angular!
        if mcurve.animCurveType == oma.MFnAnimCurve.kAnimCurveTA:
            value = om.MAngle.internalToUI(value)

        values.append(value)
    
    # Fighting precision errors :(
    rounded_values =  [round(x,10) for x in values] 
    
    return rounded_values


def get_velocity_graph(values):
    velocity_graph = [0.0]
    for i in xrange(len(values)):
        if i > 0:
            current_value = values[i]
            previous_value = values[i-1]
            velocity_graph.append(abs(current_value - previous_value))
    return velocity_graph


def get_curve_range(mcurve):
    key_times = []
    for key_index in xrange(int(mcurve.numKeys)):
        mtime = mcurve.input(key_index) # Get (time, timeType) at keyIndex
        key_times.append(mtime.value)
        
    int_range = [x * 1.0 for x in xrange(int(min(key_times)), int(max(key_times))+1)]
    
    set_range = set(int_range)
    set_range.update(key_times)
    list_range = list(set_range)
    list_range.sort()

    return list_range

def get_curve_ranges(mcurve):
    key_times = []
    for key_index in xrange(int(mcurve.numKeys)):
        mtime = mcurve.input(key_index) # Get (time, timeType) at keyIndex
        key_times.append(mtime.value)
    range_times = []
    for i, key in enumerate(key_times):
        if key != key_times[-1]:
            key_range = key_times[i], key_times[i+1]
            range_times.append(key_range)
    return range_times

def normalize_values(values, normal=100):
    # Returns a list of values where all values will add to 100
    # Normalize it to 100
    if abs(sum(values)) > 0.0:
        mult =  normal / abs(sum(values))
        return [(x * mult) for x in values]
    else:
        # cmds.error('Curve cannot be normalized. Values are flat.')
        return [0.0 for x in values]


# http://stackoverflow.com/q/3844948/
def is_equal(lst):
    return not lst or lst.count(lst[0]) == len(lst)


def map_from_to(x,a,b,c,d):
   y=(x-a)/(b-a)*(d-c)+c
   return y


def get_other_axis(attribute):
    ''' Takes a string, 
        Replaces the X Y or Z with a list of the other two
    '''
    # Wow this is hacky wtf I'm sorry
    axis = ['X', 'Y', 'Z']
    attr = [x for x in axis if x in attribute]
    axis.remove(attr[0])
    return [attribute.replace(attr[0], axis[0]), 
            attribute.replace(attr[0], axis[-1])]


def get_animated_attributes(node):
    import maya.OpenMaya as om1
    import maya.OpenMayaAnim as oma1

    # Get a MDagPath for the given node name:
    # node = 'pCube1'
    selList = om1.MSelectionList()
    selList.add(node)
    mDagPath = om1.MDagPath()
    selList.getDagPath(0, mDagPath)

    # Find all the animated attrs:
    mPlugArray = om1.MPlugArray()
    oma1.MAnimUtil.findAnimatedPlugs(mDagPath, mPlugArray)
    animCurves = []
    attribute_names = []

    # Find the curves ultimately connected to the attrs:
    for i in range(mPlugArray.length()):
        mPlugObj = om1.MPlug(mPlugArray[i])
        attribute_name = mPlugObj.name()
        attribute_names.append(attribute_name)
        
        # We could go on to capture all the animCurves of the plug
        # But this skips layer memberships. 
        # We would need a better way traverse the connections.
        # Perhaps this holds the key... 
        # https://discourse.techart.online/t/maya-animlayer-and-the-api/3510/4
        mObjArray = om1.MObjectArray()
        oma1.MAnimUtil.findAnimation(mPlugArray[i], mObjArray)
        for j in range(mObjArray.length()):
            depNodeFunc = om1.MFnDependencyNode(mObjArray[j])
            animCurves.append(depNodeFunc.name())
      
    # See what we found:      
    # for ac in sorted(animCurves):
    #     print ac
    return sorted(attribute_names)

def apply_values(curve, values):
    # Do the magic, do the magic!
    for index, key in enumerate(values):
        cmds.keyframe(curve, index=(index,), valueChange=key)

# --------------------------------------------------------------------------- #
# Run commands

def run():
    adjustment_layer, layers_to_process = get_layers_to_process() # Validates layer selection
    sel = cmds.ls(sl=1) 
    ctrl_curves_to_process = Vividict()
    for obj in sel:
        curves_to_process = get_curves_to_process(obj, adjustment_layer, layers_to_process)
        ctrl_curves_to_process[obj] = curves_to_process

    # curve_data = Vividict()
    for obj in ctrl_curves_to_process.keys():
        attributes_to_delete = []
        for attribute, layers in ctrl_curves_to_process[obj].items():
            layer_composite = []

            adjustment_curve  = layers[adjustment_layer][0]
            adjustment_range  = get_curve_range(return_MFnAnimCurve(adjustment_curve))
            adjustment_ranges = get_curve_ranges(return_MFnAnimCurve(adjustment_curve))

            for layer, curve in layers.items():
                api_curve = return_MFnAnimCurve(curve[0])
                value_graph = get_value_graph(api_curve, adjustment_range)
                
                if layer == adjustment_layer:
                    if is_equal(value_graph):
                        attributes_to_delete.append(attribute)
                        continue
                        
                    ctrl_curves_to_process[obj][attribute]['adjustment_curve']  = curve[0]
                    ctrl_curves_to_process[obj][attribute]['adjustment_layer']  = layer
                    ctrl_curves_to_process[obj][attribute]['adjustment_range']  = adjustment_range
                    ctrl_curves_to_process[obj][attribute]['adjustment_ranges']  = adjustment_ranges
                    ctrl_curves_to_process[obj][attribute]['adjustment_values'] = value_graph

                else:    
                    velocity_graph = get_velocity_graph(value_graph)
                    layer_composite.append(velocity_graph)

            # Composite the layers together
            for i, value in enumerate(layer_composite):
                if i != 0:
                    for x,_ in enumerate(value):
                        layer_composite[0][x] += value[x]
            
            ctrl_curves_to_process[obj][attribute]['composite_velocity'] = layer_composite[0]
            
            # Normalize it for final consumption
            # normalized_velocity_graph = normalize_values(layer_composite[0])
            # ctrl_curves_to_process[obj][attribute]['composite_velocity_normalized'] = normalized_velocity_graph
        
        # Skip the attributes that cannot be 'adjusted'
        for attr in attributes_to_delete:
            del ctrl_curves_to_process[obj][attr]
        

    print "Running operation on \n{}".format('\n'.join(ctrl_curves_to_process[obj].keys()))
    for obj in ctrl_curves_to_process.keys():
        for attribute, data in ctrl_curves_to_process[obj].items():
            adjustment_curve = data['adjustment_curve']
            adjustment_layer = data['adjustment_layer']
            adjustment_range = data['adjustment_range']
            adjustment_ranges = data['adjustment_ranges']
            adjustment_values = data['adjustment_values']
            composite_velocity_graph  = data['composite_velocity']
            # normalized_velocity_graph = data['composite_velocity_normalized']

            if is_equal(composite_velocity_graph):
                continue # SKIP IT for now - we don't have the clever shit installed

            new_value_curve = []
            frame_march = []
            for frange in adjustment_ranges:
        
                frame_range = range(int(frange[0]), int(frange[1])+1) 
                
                normalized_velocity_graph = normalize_values(composite_velocity_graph[adjustment_range.index(frange[0]):adjustment_range.index(frange[1])+1])
                
                ''' # This begins the 'clever' shit
                if not normalized_velocity_graph:
                    # Get neighboring axis and composite them
                    for attr in ATTRIBUTES:
                        if attr in attribute:
                            current_axis = attr
                            other_axis = get_other_axis(current_axis)
                            velocity_curve = []
                            for axis in other_axis:
                                new_axis = attribute.replace(current_axis, axis)
                                values = ctrl_curves_to_process[obj][new_axis]['composite_velocity_normalized']
                                velocity_curve.append(values)
                            # Instead of compositing the two velocities, I should check which velocity curve is more
                            # 'energetic' or has the biggest spikes of value change
                            summary_velocity = [a + b for a,b in zip(velocity_curve[0],velocity_curve[-1])]
                            normalized_velocity_graph = normalize_values(summary_velocity)
                '''
                # TODO: We need to implement a neighbor-based method of adjustment
                # if not normalized_velocity_graph:
                #     continue 
                    
                sum_percentage = 0.0
                
                for index, value in enumerate(frame_range):
                    sum_percentage += normalized_velocity_graph[index]
                    new_value = map_from_to(sum_percentage, 0, 100, adjustment_values[adjustment_range.index(frange[0])], adjustment_values[adjustment_range.index(frange[1])])
                    if value not in frame_march:
                        new_value_curve.append(new_value)
                        frame_march.append(value) # I do this to skip the repeat frames between sets - those keys already exist anyway
           
                
            # Now set the keys
            # Do the magic, DO THE MAGIC!
            if DO_SET:
                for index, time in enumerate(adjustment_range):
                    cmds.setKeyframe(adjustment_curve, animLayer=adjustment_layer, time=(time,), value=new_value_curve[index])


# --------------------------------------------------------------------------- #
# Bunch of dev shit here
def get_curve_intensity(curve):
    
    num_reversals = len(get_reversals(curve))
    
    # draw a straight line from beginning to end
    # Every time you get a reversal on the top side, it is a peak
    peaks, valleys = get_peaks_valleys(curve) 
    num_peaks = len(peaks)
    num_valleys = len(valleys)
    # how big are the peaks vs valleys?
    highest_value = max(peaks)
    lowest_value = min(valleys)
    
    # hottest moment?
    highest_velocity = max(get_velocity_graph(curve))

    return data


def compare_curve_intensities(curve1, curve2):
    # Counts the number of signals data1 beats over data2
    # Returns the winning curve
    data1 = get_curve_intensity(curve1)
    data2 = get_curve_intensity(curve2)
    
    data1_winnings = []
    for k in data1.keys():
        data1_winner = data1[k] > data2[k]
        data1_winnings.append(data1_winner)
    if data1_winnings.count(True) > data1_winnings.count(False):
        return curve1
    else:
        return curve2


# --------------------------------------------------------------------------- #
# Developer section

if __name__ == '__main__':
    run()
