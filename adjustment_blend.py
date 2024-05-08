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

DO_SET     = True
SMART      = False
DEBUG      = True

CHANNELS   = [ 'translate'
             , 'rotate'
             , 'scale'
             ]
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
GRAPH_EDITOR = 'graphEditor1GraphEd'


# if DEBUG:
#     from pprint import pprint as pp

# Helper dict will create a new key if it doesn't already exist
class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value


def get_selected_animLayers():
    '''
    Return the names of the layers which are selected
    '''
    # Get selected animLayers - OLD VERSION
    # selected_layers = cmds.treeView("AnimLayerTabanimLayerEditor", q=True, selectItem=True) or []

    layers = list()
    for each in cmds.ls(type='animLayer'):
        if cmds.animLayer(each, query=True, selected=True):
            layers.append(each)
    return layers

def get_layers_to_process():
    ''' Returns a list of animation layers that will be processed
    '''

    all_layers = []
    root_layer = cmds.animLayer(query=True, root=True)
    if root_layer:
        # This doesn't work because of nested layers. Thanks maya
        # all_layers.extend(cmds.animLayer(root_layer, q=True, children=True) or []) # top-down is last-to-first
        # This doesn't work because the layers aren't returned in their order of addition.
        # all_layers.append(root_layer)
        # all_layers.extend([x for x in cmds.ls(type='animLayer') if x != root_layer])

        # Is this really safe? Thanks maya
        all_layers = cmds.treeView("AnimLayerTabanimLayerEditor", q=True, children=True)

    else:
        return None

    # all_ls_layers = cmds.ls(type='animLayer')

    # Get selected animLayers
    selected_layers = get_selected_animLayers()


    layers_to_process = all_layers[:] # Copy the list
    if len(selected_layers) == 0 or selected_layers[-1] == 'BaseAnimation': # Why select base?? Treat it like selecting nothing, I guess.
        selected_layers = [layers_to_process.pop()] # Make it a list
    elif len(selected_layers) == 1:
        index = layers_to_process.index(selected_layers[-1])
        del layers_to_process[index:]
    elif len(selected_layers) > 1:
        layers_to_process = selected_layers[:-1]

    adjustment_layer = selected_layers[-1]

    if not isinstance(layers_to_process, list): layers_to_process = [layers_to_process]

    layers_to_remove = []
    for layer in layers_to_process:
        if cmds.animLayer(layer, q=True, lock=True):
            layers_to_remove.append(layer)

    for layer in layers_to_remove:
        layers_to_process.remove(layer)

    if DEBUG:
        print('adjustment layer target is {0}.\nSummed layers are {1}\nLocked layers are {2}'.format(selected_layers[-1], layers_to_process, layers_to_remove))


    return adjustment_layer, layers_to_process


def get_attribute_curves(attribute, layer):
    ''' Returns dict
    {animationLayer : curveName}
    '''

    obj, attr = attribute.split('.')

    # Filter out non ATTRIBUTES
    if attr not in ATTRIBUTES:
        return None

    if attribute in cmds.animLayer(layer, q=True, attribute=True) or []:
        anim_curve = cmds.animLayer(layer, q=True, findCurveForPlug=attribute) or []

    if anim_curve:
        return {layer : anim_curve[0]}


def working_scale():
    # Maya ALWAYS works in CM. So scale internals by the working units in order to get the desired result!
    units = cmds.currentUnit(q=True, linear=True)
    if units == 'mm':
        return 0.1
    if units == 'cm':
        return 1
    if units == 'm':
        return 100
    if units == 'km':
        return 1000
    if units == 'in':
        return 2.54
    if units == 'ft':
        return 30.48
    if units == 'yd':
        return 91.44
    if units == 'mi':
        return 160934


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
        else:
            # Translation curves are scaled by the WORKING UNITS! WTF?!
            value = value / working_scale()

        values.append(value)

    # Fighting precision errors :(
    rounded_values =  [round(x,10) for x in values]

    return rounded_values


def get_velocity_graph(values):
    velocity_graph = [0.0]
    for i in range(len(values)):
        if i > 0:
            current_value = values[i]
            previous_value = values[i-1]
            velocity_graph.append(abs(current_value - previous_value))
    return velocity_graph

def get_float_range(list_of_keys):

    float_range = [x * 1.0 for x in range(int(min(list_of_keys)), int(max(list_of_keys))+1)]

    set_range = set(float_range)
    set_range.update(list_of_keys)
    list_range = list(set_range)
    list_range.sort()

    return list_range

def get_curve_range(mcurve):
    key_times = []
    for key_index in range(int(mcurve.numKeys)):
        mtime = mcurve.input(key_index) # Get (time, timeType) at keyIndex
        key_times.append(mtime.value)

    int_range = [x * 1.0 for x in range(int(min(key_times)), int(max(key_times))+1)]

    set_range = set(int_range)
    set_range.update(key_times)
    list_range = list(set_range)
    list_range.sort()

    return list_range

def get_curve_ranges(mcurve):
    key_times = []
    for key_index in range(int(mcurve.numKeys)):
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


def remap(old_value, old_min, old_max, new_min, new_max):
    old_range = (old_max - old_min)

    if old_range == 0:
        new_value = new_min
    else:
        new_range = (new_max - new_min)
        new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    return new_value


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

def get_other_channel(channel):
    ''' Takes a string,
        Replaces the channel with a list of the other two
    '''
    # Wow this is hacky wtf I'm sorry
    channels = ['translate', 'rotate', 'scale']
    attr = [x for x in channels if x in channel]
    channels.remove(attr[0])
    return channels


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
        cmds.keyframe(curve, index=(index,), valueChange=key, absolute=True)

def keywithmaxval(d): # https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
    """ a) create a list of the dict's keys and values;
        b) return the key with the max value"""
    v=list(d.values())
    k=list(d.keys())
    if is_equal(v):
        return None
    else:
        return k[v.index(max(v))]




def get_attribute_layer_curve(attribute, layer):
    """ Find the curve for the attribute on the specified layer
        If no curve is found, return the value of the attribute (assume unkeyed)
    """

    if not is_object_in_layer(attribute, layer):
        return None

    plug = None
    if layer == cmds.animLayer(q=True, root=True):
        # For the base animation layer, traverse the chain of animBlendNodes all
        # the way to the end.  The plug will be "inputA" on that last node.
        conns = cmds.listConnections(attribute, type='animBlendNodeBase', source=True, destination=False)
        blendNode = None
        while conns:
            blendNode = conns[0]
            conns = cmds.listConnections(blendNode, type='animBlendNodeBase', source=True, destination=False)
        plug = '{0}.inputA'.format(blendNode)
        return cmds.getAttr(plug)
    else:
        # For every layer other than the base animation layer, we can just use
        # the "animLayer" command.  Unfortunately the "layeredPlug" flag is
        # broken in Python in Maya 2016, so we have to use MEL.
        cmd = 'animLayer -q -layeredPlug "{0}" "{1}"'.format(attribute, layer)
        plug = mel.eval(cmd)
        return cmds.listConnections(plug)
    # return plug


def is_object_in_layer(obj, layer):
    """ Determine if the given object is in the given animation layer.
    """
    object_layer_members = cmds.animLayer([obj], q=True, affectedLayers=True) or []
    if layer in object_layer_members:
        return True
    return False





# ---------------------------------------------------------------------------- #
# Run commands

def run(smart=SMART, do_set=DO_SET):
    # We only process one layer and it's controls (or the selected controls in it)

    # Start by getting layers and which layer is the adjustment layer
    adjustment_layer, layers_to_process = get_layers_to_process() # Validates layer selection
    if not layers_to_process:
        cmds.warning("No animation layers to process. Aborting!")
        # Eject if no layers
        return None

    adjustment_layer_members = cmds.animLayer( adjustment_layer
                                             , q=True
                                             , attribute=True)
    members = [x.split('.')[0] for x in adjustment_layer_members]
    members = list(set(members))

    if not adjustment_layer_members:
        cmds.warning("Adjustment layer {} has no members. Aborting!".format(adjustment_layer))
        # Eject if no layers
        return None

    objects = cmds.ls(sl=1)
    if not objects:
        # No selection? Fetch members of the adjustment layer
        cmds.warning("No object selected. Fetching members of {} instead.".format(adjustment_layer))
        objects = members
    else:
        if not bool(set(members) & set(objects)):
            cmds.warning("No selected objects exist in the selected adjustment layer!")
            return None

    if not objects:
        # Still no objects? Abort.
        cmds.warning("{} contains no controls. Aborting!".format(adjustment_layer))
        return None

    # Eject non-member layers from layers_to_process
    layers_to_not_process = []
    for layer in layers_to_process:
        if layer == cmds.animLayer(q=True, root=True): continue
        layer_members = cmds.animLayer( layer
                                      , q=True
                                      , attribute=True)
        members = [x.split('.')[0] for x in layer_members]
        members = list(set(members))
        if not bool(set(members) & set(objects)):
            print( "Didn't find {0} in {1}.".format(members, layer))
            layers_to_not_process.append(layer)

    for layer in layers_to_not_process:
        layers_to_process.remove(layer)



    # Validation done... sort of

    # ======================================================================= #
    # BEGIN
    # ======================================================================= #

    adjustment_keys = set()
    ctrl_curves_to_process = Vividict()
    attributes_to_skip = {}
    attributes_to_skip["No adjustment curve found"] = []
    attributes_to_skip["Adjustment curve has no change in values"] = []
    attributes_to_skip["Adjustment attribute was not keyed"] = []
    attributes_to_skip["Adjustment curve has no changing values below it"] = []

    # This section will populate the dictionary like so:
    # control
    # ? attribute
    #   ? animation layer
    #     ? animation curve OR static value

    for obj in objects:

        animated_attributes = get_animated_attributes(obj) # The API version

        for layer in layers_to_process + [adjustment_layer]:

            for attribute in animated_attributes:

                obj, attr = attribute.split('.')

                if attr not in ATTRIBUTES:
                    continue # Whitelisting attributes for now

                if attribute in adjustment_layer_members:
                    if layer == cmds.animLayer(q=True, root=True): # BaseAnimation is treated differently... thanks Maya
                        # Now we traverse the tree going from the top animLayer down to the base
                        connections = cmds.listConnections(attribute, type='animBlendNodeBase', source=True, destination=False)
                        blend_node = None
                        while connections:
                            blend_node = connections[0]
                            connections = cmds.listConnections(blend_node, type='animBlendNodeBase', source=True, destination=False)
                        plug = '{0}.inputA'.format(blend_node) # We hit base

                        curve = cmds.listConnections(plug) or []
                        if curve:
                            ctrl_curves_to_process[obj][attr][layer] = curve

                        else:
                            # Treat rotations differently on the base as well... thanks maya
                            if cmds.nodeType(blend_node) == 'animBlendNodeAdditiveRotation':
                                if 'X' in attr:
                                    plug = plug + 'X'
                                if 'Y' in attr:
                                    plug = plug + 'Y'
                                if 'Z' in attr:
                                    plug = plug + 'Z'
                                ctrl_curves_to_process[obj][attr][layer] = cmds.listConnections(plug)

                            else: # Everything else is fine
                                ctrl_curves_to_process[obj][attr][layer] = cmds.getAttr(plug)
                    else:
                        if attribute not in cmds.animLayer(layer, q=True, attribute=True):
                            # print layer
                            continue
                        plug = cmds.animLayer(layer, q=True, layeredPlug=attribute)
                        curve = cmds.animLayer(layer, q=True, findCurveForPlug=attribute)
                        if curve:
                            if layer == adjustment_layer:
                                keyframes = cmds.keyframe(curve, q=True) or []
                                for key in keyframes:
                                    adjustment_keys.add(key)

                                values = cmds.keyframe(curve, q=True, valueChange=True) or []
                                if is_equal(values):
                                    attributes_to_skip["Adjustment curve has no change in values"].append(attribute)
                                    continue
                            ctrl_curves_to_process[obj][attr][layer] = curve
                        else:
                            if layer == adjustment_layer:
                                attributes_to_skip["Adjustment attribute was not keyed"].append(attribute)
                                continue
                            # try:
                            #     ctrl_curves_to_process[obj][attr][layer] = cmds.getAttr(plug.replace('.inputB', '.inputA'))
                            # finally:
                            #     cmds.error("No input found for {0} or {1}".format(attribute, plug.replace('.inputB', '.inputA')))
                            if plug:
                                ctrl_curves_to_process[obj][attr][layer] = cmds.getAttr(plug.replace('.inputB', '.inputA'))
                            else:
                                cmds.error("No input found for {0}".format(attribute))
                                continue
        if not ctrl_curves_to_process[obj]: continue # Eject ghosts

    if not ctrl_curves_to_process: return False # How does this happen?


    # Clean out any attribute that holds no value-changing adjustment curves
    for obj in list(ctrl_curves_to_process.keys()):
        for attr in list(ctrl_curves_to_process[obj].keys()):
            attribute = obj + '.' + attr
            if attribute in attributes_to_skip:
                del ctrl_curves_to_process[obj][attr]
            if adjustment_layer not in ctrl_curves_to_process[obj][attr].keys():
                del ctrl_curves_to_process[obj][attr]


    # At this point, we have the curve names of objects on the
    # adjustment layer, and keys of all the objects on this layer.
    # So we can composite adjustment ranges between these keys

    adjustment_key_ranges = []
    adjustment_keys_sorted = sorted(adjustment_keys)
    for index, key in enumerate(adjustment_keys_sorted):
        if not index == len(adjustment_keys_sorted) - 1:
            adjustment_key_ranges.append([key, adjustment_keys_sorted[index+1]])

    # Working calculation range
    if not adjustment_keys:
        cmds.warning("Could not find any adjustment keys on {}".format(adjustment_layer))
        return False
    if len(adjustment_keys) == 1:
        cmds.warning("Could not find aenough adjustment keys on {}".format(adjustment_layer))
        return False

    calculation_range = get_float_range(adjustment_keys)

    # Now we need to calculate the layers_to_process between these ranges
    # We can query all curves between these ranges to get value graphs
    # If we don't find a curve (no key on BaseAnimation for example), we can grab the flat value

    for obj in ctrl_curves_to_process.keys():

        # get_rotates    = False
        # get_translates = False

        for attr in ctrl_curves_to_process[obj].keys():

            # if smart:
            #     if 'rotate' in attr:
            #         get_rotates = True
            #     if 'translate' in attr:
            #         get_translates = True

            for layer, destination in ctrl_curves_to_process[obj][attr].items():

                if not isinstance(destination, list):
                    destination = ctrl_curves_to_process[obj][attr][layer] = [destination]

                if isinstance(destination[0], float):
                    float_range = [destination[0] for x in calculation_range]

                elif isinstance(destination[0], str):
                    float_range = []
                    for time in calculation_range:
                        value = cmds.keyframe(destination[0], q=True, valueChange=True, eval=True, time=(time,))[0]
                        float_range.append(value)

                else:
                    cmds.error("Something went horribly wrong with {0}, {1}.".format(layer, destination))
                    continue

                if layer == adjustment_layer and is_equal(float_range):
                    # Constant values are irrelevant
                    print( "Deleting {0}.{1}".format(obj, attr))
                    del ctrl_curves_to_process[obj][attr]
                    continue

                ctrl_curves_to_process[obj][attr][layer].append(float_range)


    # ======================================================================= #
    # Begin calculation of the curve data

    value_graphs = Vividict()

    for obj in ctrl_curves_to_process.keys():

        for attr in ctrl_curves_to_process[obj].keys():

            composite_velocity_graphs = []

            for layer, destination in ctrl_curves_to_process[obj][attr].items():

                if isinstance(destination[0], float):
                    continue

                elif isinstance(destination[0], str):
                    api_curve = return_MFnAnimCurve(destination[0])
                    value_graph = get_value_graph(api_curve, calculation_range)

                    if is_equal(value_graph):
                        continue # Constant values are irrelevant

                    if layer == adjustment_layer:
                        value_graphs[obj][attr]['adjustment_graph'] = value_graph
                        value_graphs[obj][attr]['adjustment_curve'] = destination[0]
                    else:
                        composite_velocity_graphs.append(get_velocity_graph(value_graph))

            if composite_velocity_graphs:
                for graph in composite_velocity_graphs:
                    for i, value in enumerate(composite_velocity_graphs):
                        if i != 0:
                            for x,_ in enumerate(value):
                                composite_velocity_graphs[0][x] += value[x]
                value_graphs[obj][attr]['composite_graph'] = composite_velocity_graphs[0]
            else:
                attributes_to_skip["Adjustment curve has no changing values below it"].append(obj + '.' + attr)
                if not smart:
                    try: # Gotta figure this out when it comes to SMARTS
                        del value_graphs[obj][attr]
                    except: pass

    adjustment_range = range(int(adjustment_key_ranges[0][0]), int(adjustment_key_ranges[-1][-1])+1)

    if DEBUG:
        any_values = bool(len(['' for x in attributes_to_skip.values() if x]))
        if any_values:
            cmds.warning("Ejected the following attributes:")
        for reason in attributes_to_skip.keys():
            if attributes_to_skip[reason]:
                print( "# " + reason)
                for attr in attributes_to_skip[reason]:
                    print( "  - " + attr)

    for obj in value_graphs.keys():
        # Just in case this was sanitized earlier
        if not value_graphs[obj].keys():
            cmds.warning("No adjustment possible for {}".format(obj))
            continue

        for attr in value_graphs[obj].keys():
            adjustment_curve = value_graphs[obj][attr]['adjustment_curve']
            adjustment_graph = value_graphs[obj][attr]['adjustment_graph']
            composite_graph  = value_graphs[obj][attr]['composite_graph']

            if not composite_graph or not adjustment_curve or not adjustment_graph:
                if not composite_graph:
                    # Need to look at adjacent axis to borrow a composite graph.

                    axis1, axis2 = get_other_axis(attr)
                    # print("comparing {} to {} and {}".format(attr, axis1, axis2) )
                    # print(value_graphs[obj][axis1]['composite_graph'])
                    # print(value_graphs[obj][axis2]['composite_graph'])

                    axis1compare = value_graphs[obj][axis1]['adjustment_curve']
                    axis2compare = value_graphs[obj][axis2]['adjustment_curve']
                    highest_intensity_curve = compare_curve_intensities(axis1compare, axis2compare)

                    if highest_intensity_curve == value_graphs[obj][axis1]['adjustment_curve']:
                        # print("substituting {0} for {1}".format(attr, axis1))
                        composite_graph = value_graphs[obj][axis1]['composite_graph']
                    elif highest_intensity_curve == value_graphs[obj][axis2]['adjustment_curve']:
                        composite_graph = value_graphs[obj][axis2]['composite_graph']
                        # print("substituting {0} for {1}".format(attr, axis2))
                    if not composite_graph:
                        # Looks like no suitable composite graph was found. Extending search to other channel (ie, rotate to translate).
                        # print("Attr {} has failed at finding a suitable composite graph.".format(attr))

                        # channel1, channel2 = get_other_channel(attr)
                        composite_graph_compare = Vividict()
                        for channel in get_other_channel(attr):
                            for axis in ['X', 'Y', 'Z']:
                                if channel+axis in value_graphs[obj].keys():
                                    values = value_graphs[obj][channel+axis]['composite_graph']
                                    composite_graph_compare[channel+axis] = max(get_velocity_graph(values))

                        hottest = keywithmaxval(composite_graph_compare)
                        if hottest:
                            # print("found hottest channel as {}".format(hottest))
                            composite_graph = value_graphs[obj][hottest]['composite_graph']
                            # print("hottest composite graph is {}".format(composite_graph))
                            print("substituting {0} for {1}".format(attr, hottest))
                # continue

            new_value_curve = []
            frame_march = []

            for frange in adjustment_key_ranges:


                frame_range = range(int(frange[0]), int(frange[1])+1)

                normalized_velocity_graph = normalize_values(composite_graph[calculation_range.index(frange[0]):calculation_range.index(frange[1])+1])

                # if is_equal(normalized_velocity_graph): continue # How did this end up here?

                sum_percentage = 0.0

                for index, value in enumerate(frame_range):
                    sum_percentage += normalized_velocity_graph[index]
                    new_value = map_from_to(sum_percentage, 0, 100, adjustment_graph[calculation_range.index(frange[0])], adjustment_graph[calculation_range.index(frange[1])])
                    if value not in frame_march:
                        new_value_curve.append(new_value)
                        frame_march.append(value) # I do this to skip the repeat frames between sets - those keys already exist anyway

            # Now set the keys
            # Do the magic, DO THE MAGIC!
            if do_set:
                # if DEBUG:
                #     print "Running adjustment on {}.".format(adjustment_curve)
                for index, time in enumerate(adjustment_range):
                    cmds.setKeyframe(adjustment_curve, animLayer=adjustment_layer, time=(time,), value=new_value_curve[index])

    if DEBUG:
        # To check whether the dict has any non-zero length value in it (returns True or False):
        any_values = bool(len(['' for x in value_graphs.values() if x]))
        if any_values:
            cmds.warning("Executing adjustment of the following attributes:")
        for obj, attr in value_graphs.items():
            for at in attr:
                print( "  + " + obj + '.' + at)

    if not do_set:
        cmds.warning("Skipped do_set. Hopefully you have DEBUG on?")



# ---------------------------------------------------------------------------- #
# Bunch of dev shit here

def num_reversals(values):
    reverals = []
    begin = False
    falling = False

    for index, value in enumerate(values):
        if index == 0: # ignore first key
            continue

        if value == values[index-1]: # ignore redunant keys
            continue

        # First direction change
        if begin == False:
            if value < values[index-1]:
                reverals.append(values[index-1])
                falling = True
            elif value > values[index-1]:
                reverals.append(values[index-1])
                falling = False
            begin = True
            continue

        if value < values[index-1] and falling == False:
            reverals.append(values[index-1])
            falling = True
            # continue
        elif value > values[index-1] and falling == True:
            reverals.append(values[index-1])
            falling = False
        continue
    return reverals


def get_peaks_valleys(curve, frange=None):
    if isinstance(curve, str):
        mcurve = return_MFnAnimCurve(curve)
    elif isinstance(curve, oma.MFnAnimCurve):
        mcurve = curve
    else:
        cmds.error("Could not fetch curve from {}".format(curve))
        return None

    if not frange:
        frange = get_curve_range(mcurve)

    frame_difference = frange[-1] - frange[0]
    frame_difference = 1 if frame_difference == 0 else frame_difference

    value_graph = get_value_graph(mcurve)
    value_graph_times = []
    for index in range(mcurve.numKeys):
        time = mcurve.input(index)
        value_graph_times.append(time.value)

    # Skewing to right to match left value
    value_graph_skewed = skew_curve(curve)


def skew_values(values):
    frame_difference = len(values) - 1
    frame_difference = 1 if frame_difference == 0 else frame_difference

    offset_value = values[-1] - values[0] # The difference from first to last frame

    value_graph_skewed = []
    for index, value in enumerate(values):
        # frame = frange[index]

        time_slope = 1 - ((index - 1) / frame_difference) # Count from 1.0 to 0.0
        pivot_value = value - offset_value
        # Basically, just multiply it by the offset then multiply THAT by how far down the frange we are
        new_value = ((value - pivot_value) * time_slope) + pivot_value

        value_graph_skewed.append(new_value)

    return value_graph_skewed


def skew_curve(curve, frange=None):
    if isinstance(curve, str):
        mcurve = return_MFnAnimCurve(curve)
    elif isinstance(curve, oma.MFnAnimCurve):
        mcurve = curve
    else:
        cmds.error("Could not fetch curve from {}".format(curve))
        return None

    if not frange:
        frange = get_curve_range(mcurve)

    frame_difference = frange[-1] - frange[0]
    frame_difference = 1 if frame_difference == 0 else frame_difference

    value_graph = get_value_graph(mcurve)

    first_value = mcurve.value(0)
    last_value = mcurve.value(mcurve.numKeys - 1)
    # offset_value = first_value - last_value # The difference from first to last frame
    offset_value = last_value - first_value # The difference from first to last frame

    value_graph_skewed = []
    for index, value in enumerate(value_graph):
        frame = frange[index]

        time_slope = 1 - ((frame - frange[0]) / frame_difference) # Count from 1.0 to 0.0
        pivot_value = value - offset_value
        # Basically, just multiply it by the offset then multiply THAT by how far down the frange we are
        new_value = ((value - pivot_value) * time_slope) + pivot_value

        value_graph_skewed.append(new_value)

    return value_graph_skewed

def get_curve_intensity(curve):
    # print("getting curve intensity of {}".format(curve))

    if isinstance(curve, str):
        mcurve = return_MFnAnimCurve(curve)
    elif isinstance(curve, oma.MFnAnimCurve):
        mcurve = curve
    else:
        cmds.error("Could not fetch curve from {}".format(curve))
        return None

    curve_data = {}

    value_graph = get_value_graph(mcurve)
    velocity_graph = get_velocity_graph(value_graph)

    value_graph_skewed = skew_curve(curve)
    reversals = num_reversals(value_graph_skewed)

    pivot_value = value_graph_skewed[0]
    peaks = []
    valleys = []
    redundants = 0.0
    if peaks and valleys:
        for point in reversals:
            if point > pivot_value:
                peaks.append(point)
            elif point < pivot_value:
                valleys.append(point)

        for index, value in enumerate(velocity_graph):
            if index == 0: continue
            if value == velocity_graph[index - 1]:
                redundants += 1


        # draw a straight line from beginning to end
        # Every time you get a reversal on the top side, it is a peak
        num_peaks = len(peaks)
        num_valleys = len(valleys)
        # how big are the peaks vs valleys?
        highest_value = max(peaks)
        lowest_value = min(valleys)

    # hottest moment?
    highest_velocity = max(velocity_graph)
    total_change = sum(velocity_graph)

    # roll it into a data set
    curve_data['redundants']       = redundants
    # curve_data['num_peaks']      = num_peaks
    # curve_data['total_change']   = total_change
    # curve_data['num_valleys']    = num_valleys
    # curve_data['lowest_value']   = lowest_value
    # curve_data['num_reversals']  = len(reversals)
    # curve_data['highest_value']  = highest_value
    curve_data['highest_velocity'] = highest_velocity


    return curve_data


def compare_graph_intensities(graph1, graph2):
    data1 = max(get_velocity_graph(graph1))
    data2 = max(get_velocity_graph(graph2))
    if data1 > data2:
        return data1
    elif data1 < data2:
        return data2
    else:
        return None

def compare_curve_intensities(curve1, curve2):
    # Counts the number of signals data1 beats over data2
    # Returns the winning curve
    data1 = get_curve_intensity(curve1)
    data2 = get_curve_intensity(curve2)
    # winner = max(data1['highest_velocity'], data2['highest_velocity'])

    if data1["highest_velocity"] > data2["highest_velocity"]:
        return curve1
    else:
        return curve2


def get_selected_curves():
    # get the key selection
    if not cmds.animCurveEditor(GRAPH_EDITOR, exists=True):
        cmds.error("No GraphEditor found.")
        return # Cannot find graph editor?

    if not cmds.animCurveEditor(GRAPH_EDITOR, q=True, areCurvesSelected=True):
        cmds.warning("Must select some keys to fit.")
        return

    selected_curves = cmds.keyframe(q=True, selected=True, name=True) or []

    return selected_curves


def get_curve_data():
    curves = get_selected_curves()
    anim_data = {}
    all_frames = []
    for curve in curves:
        selected_frames = cmds.keyframe(curve, q=True, selected=True, timeChange=True)
        all_frames.extend(selected_frames)

        # selected_index = cmds.keyframe(curve, q=True, selected=True, indexValue=True)
        selected_values = cmds.keyframe(curve, q=True, selected=True, valueChange=True)
        anim_data[curve] = [selected_frames, selected_values]

    first_frame = min(all_frames)
    last_frame = max(all_frames)


# ---------------------------------------------------------------------------- #
# Developer section

if __name__ == '__main__':
    print("# " + 76*"=" + " #\n") # Divider
    run(smart=True, do_set=True)
    # pass
