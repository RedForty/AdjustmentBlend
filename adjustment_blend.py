"""
Adjustment Blend - Velocity-matched keyframe interpolation for Maya animation layers.

This tool generates keyframes between sparse keyframes on adjustment layers,
matching the rate-of-change (velocity) from the layers below instead of using
linear or bezier interpolation.

Based on Dan Low's GDC talk on animation blending.

Architecture:
    1. gather_context() - Identifies layers, objects, and frame ranges
    2. collect_attribute_data() - Uses animLib.LayerStack for composite values
    3. calculate_adjustment_values() - Distributes values by velocity
    4. apply_keyframes() - Writes keyframes to Maya

Author: Daniel Klug
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set
import logging

from maya import cmds, mel
import maya.api.OpenMaya as om
import maya.api.OpenMayaAnim as oma

# Import animLib for LayerStack functionality
import animLib

# =============================================================================
# Configuration
# =============================================================================

DO_SET = True
SMART = False
DEBUG = True

CHANNELS = ['translate', 'rotate', 'scale']
ATTRIBUTES = [
    'translateX', 'translateY', 'translateZ',
    'rotateX', 'rotateY', 'rotateZ',
    'scaleX', 'scaleY', 'scaleZ',
]
GRAPH_EDITOR = 'graphEditor1GraphEd'

# Logging setup
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG if DEBUG else logging.WARNING)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AdjustmentContext:
    """
    Context for an adjustment blend operation.

    Contains all the "what to process" information gathered at the start.
    """
    adjustment_layer: str
    layers_to_process: List[str]
    objects: List[str]
    adjustment_keys: List[float]
    adjustment_key_ranges: List[tuple]  # [(start, end), ...]
    calculation_range: List[float]  # All frames to evaluate


@dataclass
class AttributeData:
    """
    Data for a single object.attribute being processed.

    This replaces the nested Vividict dictionary with explicit, typed fields.
    """
    obj: str
    attr: str
    adjustment_curve: str
    adjustment_values: List[float]  # Values at each frame in calculation_range
    composite_velocity: List[float]  # Velocity graph from layers below

    # Fallback tracking (for smart mode)
    velocity_source: str = ""  # Which attr the velocity came from (for debugging)

    @property
    def full_attr(self) -> str:
        return f"{self.obj}.{self.attr}"

    def has_valid_velocity(self) -> bool:
        """Check if the composite velocity has any motion."""
        if not self.composite_velocity:
            return False
        return not is_equal(self.composite_velocity)


# Legacy helper - kept for compatibility with existing code
class Vividict(dict):
    """Auto-vivifying dictionary. DEPRECATED - use data classes instead."""
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





# =============================================================================
# Pipeline Stage 1: Gather Context
# =============================================================================

def gather_context(objects: Optional[List[str]] = None) -> Optional[AdjustmentContext]:
    """
    Gather all context needed for adjustment blend operation.

    This is the first pipeline stage - it validates the scene state and
    returns a context object with all the "what to process" information.

    Args:
        objects: Optional list of objects to process. If None, uses selection
                 or adjustment layer members.

    Returns:
        AdjustmentContext if valid, None if validation fails.
    """
    # Get layer configuration
    layer_result = get_layers_to_process()
    if not layer_result:
        cmds.warning("No animation layers to process. Aborting!")
        return None

    adjustment_layer, layers_to_process = layer_result

    if not layers_to_process:
        cmds.warning("No animation layers below adjustment layer to process. Aborting!")
        return None

    # Get adjustment layer members
    adjustment_layer_members = cmds.animLayer(adjustment_layer, q=True, attribute=True) or []
    if not adjustment_layer_members:
        cmds.warning(f"Adjustment layer {adjustment_layer} has no members. Aborting!")
        return None

    members = list(set(x.split('.')[0] for x in adjustment_layer_members))

    # Determine objects to process
    if objects is None:
        objects = cmds.ls(sl=1) or []

    if not objects:
        cmds.warning(f"No object selected. Fetching members of {adjustment_layer} instead.")
        objects = members
    else:
        # Validate selected objects are in adjustment layer
        if not bool(set(members) & set(objects)):
            cmds.warning("No selected objects exist in the selected adjustment layer!")
            return None

    if not objects:
        cmds.warning(f"{adjustment_layer} contains no controls. Aborting!")
        return None

    # Filter layers that don't contain our objects
    layers_to_process = _filter_layers_by_objects(layers_to_process, objects)

    # Collect all adjustment keys from the adjustment layer
    adjustment_keys = _collect_adjustment_keys(objects, adjustment_layer, adjustment_layer_members)

    if not adjustment_keys:
        cmds.warning(f"Could not find any adjustment keys on {adjustment_layer}")
        return None

    if len(adjustment_keys) < 2:
        cmds.warning(f"Need at least 2 adjustment keys on {adjustment_layer}")
        return None

    # Build key ranges (segments between keys)
    adjustment_keys_sorted = sorted(adjustment_keys)
    adjustment_key_ranges = [
        (adjustment_keys_sorted[i], adjustment_keys_sorted[i + 1])
        for i in range(len(adjustment_keys_sorted) - 1)
    ]

    # Build calculation range (all frames to evaluate)
    calculation_range = get_float_range(list(adjustment_keys))

    if DEBUG:
        log.debug(f"Adjustment layer: {adjustment_layer}")
        log.debug(f"Layers to process: {layers_to_process}")
        log.debug(f"Objects: {objects}")
        log.debug(f"Key ranges: {adjustment_key_ranges}")

    return AdjustmentContext(
        adjustment_layer=adjustment_layer,
        layers_to_process=layers_to_process,
        objects=objects,
        adjustment_keys=adjustment_keys_sorted,
        adjustment_key_ranges=adjustment_key_ranges,
        calculation_range=calculation_range,
    )


def _filter_layers_by_objects(layers: List[str], objects: List[str]) -> List[str]:
    """Remove layers that don't contain any of the target objects."""
    root_layer = cmds.animLayer(q=True, root=True)
    filtered = []

    for layer in layers:
        if layer == root_layer:
            filtered.append(layer)
            continue

        layer_members = cmds.animLayer(layer, q=True, attribute=True) or []
        layer_objects = set(x.split('.')[0] for x in layer_members)

        if bool(layer_objects & set(objects)):
            filtered.append(layer)
        else:
            log.debug(f"Skipping layer {layer} - no target objects")

    return filtered


def _collect_adjustment_keys(
    objects: List[str],
    adjustment_layer: str,
    adjustment_layer_members: List[str]
) -> Set[float]:
    """Collect all keyframe times from adjustment layer for target objects."""
    adjustment_keys = set()

    for obj in objects:
        animated_attributes = get_animated_attributes(obj)

        for attribute in animated_attributes:
            obj_name, attr = attribute.split('.')
            if attr not in ATTRIBUTES:
                continue

            if attribute not in adjustment_layer_members:
                continue

            curve = cmds.animLayer(adjustment_layer, q=True, findCurveForPlug=attribute)
            if curve:
                keyframes = cmds.keyframe(curve, q=True) or []
                adjustment_keys.update(keyframes)

    return adjustment_keys


# =============================================================================
# Pipeline Stage 2: Collect Attribute Data (using animLib.LayerStack)
# =============================================================================

def collect_attribute_data(context: AdjustmentContext) -> List[AttributeData]:
    """
    Collect attribute data using animLib.LayerStack for composite values.

    This is the key integration point with animLib - instead of manually
    traversing blend nodes, we use LayerStack.get_base_values() to get
    the composite values from all layers below the adjustment layer.

    Args:
        context: The adjustment context from gather_context()

    Returns:
        List of AttributeData objects ready for velocity calculation.
    """
    attribute_data_list = []
    skipped = {
        "no_adjustment_curve": [],
        "flat_adjustment": [],
        "no_composite_velocity": [],
    }

    adjustment_layer_members = set(
        cmds.animLayer(context.adjustment_layer, q=True, attribute=True) or []
    )

    for obj in context.objects:
        # Build LayerStack for this object - this is the animLib integration!
        # The LayerStack knows how to traverse blend nodes and get composite values
        try:
            stack = animLib.LayerStack.build(obj, context.adjustment_layer)
        except Exception as e:
            log.warning(f"Could not build LayerStack for {obj}: {e}")
            continue

        animated_attributes = get_animated_attributes(obj)

        for attribute in animated_attributes:
            _, attr = attribute.split('.')

            if attr not in ATTRIBUTES:
                continue

            if attribute not in adjustment_layer_members:
                continue

            # Get the adjustment curve
            curve = cmds.animLayer(
                context.adjustment_layer, q=True, findCurveForPlug=attribute
            )
            if not curve:
                skipped["no_adjustment_curve"].append(attribute)
                continue

            curve = curve[0] if isinstance(curve, list) else curve

            # Get adjustment values at each frame
            adjustment_values = []
            for t in context.calculation_range:
                value = cmds.keyframe(curve, q=True, valueChange=True, eval=True, time=(t,))
                adjustment_values.append(value[0] if value else 0.0)

            # Skip if adjustment curve is flat
            if is_equal(adjustment_values):
                skipped["flat_adjustment"].append(attribute)
                continue

            # Get composite values from layers below using LayerStack
            # This replaces the manual blend node traversal!
            composite_values = stack.get_base_values(attr, context.calculation_range)

            # Calculate velocity from composite values
            composite_velocity = get_velocity_graph(composite_values)

            attr_data = AttributeData(
                obj=obj,
                attr=attr,
                adjustment_curve=curve,
                adjustment_values=adjustment_values,
                composite_velocity=composite_velocity,
                velocity_source=attr,
            )

            if not attr_data.has_valid_velocity():
                skipped["no_composite_velocity"].append(attribute)
                # Still add it - we might find a fallback in smart mode
                # The velocity will be filled in by apply_smart_fallbacks()

            attribute_data_list.append(attr_data)

    if DEBUG:
        for reason, attrs in skipped.items():
            if attrs:
                log.debug(f"Skipped ({reason}): {attrs}")

    return attribute_data_list


# =============================================================================
# Pipeline Stage 3: Calculate Adjustment Values
# =============================================================================

def calculate_adjustment_values(
    attr_data: AttributeData,
    context: AdjustmentContext
) -> List[float]:
    """
    Calculate new keyframe values for an attribute based on velocity distribution.

    This is the core algorithm: instead of linear interpolation between
    adjustment keyframes, we distribute the value change according to
    where motion is happening in the layers below.

    Args:
        attr_data: The attribute data with velocity information
        context: The adjustment context

    Returns:
        List of new values for each frame in the adjustment range.
    """
    new_values = []
    frame_march = set()

    for start_frame, end_frame in context.adjustment_key_ranges:
        frame_range = range(int(start_frame), int(end_frame) + 1)

        # Get the velocity slice for this range
        start_idx = context.calculation_range.index(start_frame)
        end_idx = context.calculation_range.index(end_frame) + 1
        velocity_slice = attr_data.composite_velocity[start_idx:end_idx]

        # Normalize velocity to 0-100%
        normalized_velocity = normalize_values(velocity_slice)

        # Get adjustment values at start and end of range
        adj_start = attr_data.adjustment_values[start_idx]
        adj_end = attr_data.adjustment_values[end_idx - 1]

        # Accumulate velocity and map to adjustment value range
        sum_percentage = 0.0

        for i, frame in enumerate(frame_range):
            sum_percentage += normalized_velocity[i]

            try:
                new_value = map_from_to(sum_percentage, 0, 100, adj_start, adj_end)
            except (TypeError, ZeroDivisionError):
                new_value = adj_start

            # Skip duplicate frames between ranges
            if frame not in frame_march:
                new_values.append(new_value)
                frame_march.add(frame)

    return new_values


# =============================================================================
# Pipeline Stage 4: Apply Keyframes
# =============================================================================

def apply_keyframes(
    attr_data: AttributeData,
    new_values: List[float],
    context: AdjustmentContext
) -> bool:
    """
    Apply calculated keyframe values to the adjustment curve.

    Args:
        attr_data: The attribute data
        new_values: The new values to apply
        context: The adjustment context

    Returns:
        True if successful, False otherwise.
    """
    adjustment_range = range(
        int(context.adjustment_key_ranges[0][0]),
        int(context.adjustment_key_ranges[-1][1]) + 1
    )

    try:
        for i, frame in enumerate(adjustment_range):
            if i >= len(new_values):
                break
            cmds.setKeyframe(
                attr_data.adjustment_curve,
                animLayer=context.adjustment_layer,
                time=(frame,),
                value=new_values[i]
            )
        return True
    except Exception as e:
        log.error(f"Failed to apply keyframes for {attr_data.full_attr}: {e}")
        return False


# =============================================================================
# Smart Fallback Logic
# =============================================================================

def apply_smart_fallbacks(
    attribute_data_list: List[AttributeData],
    context: AdjustmentContext
) -> List[AttributeData]:
    """
    Apply fallback logic to find velocity data for attributes with flat composites.

    When an attribute has no velocity in its composite graph, we try:
    1. Other axes of the same channel (translateX -> translateY, translateZ)
    2. Other channels entirely (translate -> rotate -> scale)

    Args:
        attribute_data_list: List of attribute data objects
        context: The adjustment context

    Returns:
        Updated list with fallback velocities applied.
    """
    # Build lookup for quick access
    lookup: Dict[str, AttributeData] = {
        f"{ad.obj}.{ad.attr}": ad for ad in attribute_data_list
    }

    for attr_data in attribute_data_list:
        if attr_data.has_valid_velocity():
            continue

        # Try other axes first
        fallback = _find_axis_fallback(attr_data, lookup)

        # Try other channels if axis fallback failed
        if not fallback:
            fallback = _find_channel_fallback(attr_data, lookup)

        if fallback:
            attr_data.composite_velocity = fallback.composite_velocity[:]
            attr_data.velocity_source = fallback.attr
            log.debug(f"Substituting {fallback.attr} velocity for {attr_data.attr}")

    return attribute_data_list


def _find_axis_fallback(
    attr_data: AttributeData,
    lookup: Dict[str, AttributeData]
) -> Optional[AttributeData]:
    """Find fallback velocity from other axes of the same channel."""
    other_axes = get_other_axis(attr_data.attr)

    for other_attr in other_axes:
        key = f"{attr_data.obj}.{other_attr}"
        if key in lookup and lookup[key].has_valid_velocity():
            return lookup[key]

    return None


def _find_channel_fallback(
    attr_data: AttributeData,
    lookup: Dict[str, AttributeData]
) -> Optional[AttributeData]:
    """Find fallback velocity from other channels."""
    other_channels = get_other_channel(attr_data.attr)
    best_fallback = None
    best_velocity = 0.0

    for channel in other_channels:
        for axis in ['X', 'Y', 'Z']:
            key = f"{attr_data.obj}.{channel}{axis}"
            if key in lookup and lookup[key].has_valid_velocity():
                max_vel = max(lookup[key].composite_velocity)
                if max_vel > best_velocity:
                    best_velocity = max_vel
                    best_fallback = lookup[key]

    return best_fallback


# =============================================================================
# Main Entry Point (New Pipeline)
# =============================================================================

def run_pipeline(smart: bool = SMART, do_set: bool = DO_SET) -> bool:
    """
    Run the adjustment blend using the new pipeline architecture.

    This is the recommended entry point - it uses proper data structures
    and animLib.LayerStack for composite value reading.

    Args:
        smart: If True, apply fallback logic for flat velocity graphs
        do_set: If True, actually write keyframes. If False, dry run.

    Returns:
        True if successful, False otherwise.
    """
    # Stage 1: Gather context
    context = gather_context()
    if not context:
        return False

    # Stage 2: Collect attribute data
    attribute_data_list = collect_attribute_data(context)
    if not attribute_data_list:
        cmds.warning("No attributes to process!")
        return False

    # Stage 2.5: Apply smart fallbacks if enabled
    if smart:
        attribute_data_list = apply_smart_fallbacks(attribute_data_list, context)

    # Filter out attributes with no valid velocity
    valid_attributes = [ad for ad in attribute_data_list if ad.has_valid_velocity()]

    if not valid_attributes:
        cmds.warning("No attributes have valid velocity data to process!")
        return False

    # Stage 3 & 4: Calculate and apply for each attribute
    processed = []
    for attr_data in valid_attributes:
        new_values = calculate_adjustment_values(attr_data, context)

        if do_set:
            if apply_keyframes(attr_data, new_values, context):
                processed.append(attr_data.full_attr)

    if DEBUG:
        if processed:
            log.info(f"Processed attributes: {processed}")
        if not do_set:
            cmds.warning("Skipped do_set. Dry run complete.")

    return bool(processed)


# =============================================================================
# Legacy Entry Point (Original Implementation)
# =============================================================================

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
    # for obj in list(ctrl_curves_to_process.keys()):
    #     for attr in list(ctrl_curves_to_process[obj].keys()):
    #         attribute = obj + '.' + attr
    #         if attribute in attributes_to_skip:
    #             del ctrl_curves_to_process[obj][attr]
    #         if adjustment_layer not in ctrl_curves_to_process[obj][attr].keys():
    #             del ctrl_curves_to_process[obj][attr]


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

                    # if is_equal(value_graph):
                    #     continue # Constant values are irrelevant

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

            if not composite_graph or not adjustment_curve or not adjustment_graph or is_equal(composite_graph):

                if not composite_graph or is_equal(composite_graph):
                    # Need to look at adjacent axis to borrow a composite graph.

                    axis1, axis2 = get_other_axis(attr)
                    print("comparing {} to {} and {}".format(attr, axis1, axis2))
                    if is_equal(value_graphs[obj][axis2]['composite_graph']) and not is_equal(value_graphs[obj][axis1]['composite_graph']):
                        print("substituting {0} for {1}".format(attr, axis1))
                        composite_graph = value_graphs[obj][axis1]['composite_graph']

                    if is_equal(value_graphs[obj][axis1]['composite_graph']) and not is_equal(value_graphs[obj][axis2]['composite_graph']):
                        print("substituting {0} for {1}".format(attr, axis2))
                        composite_graph = value_graphs[obj][axis2]['composite_graph']

                    # print(value_graphs[obj][axis1]['composite_graph'])
                    # print(value_graphs[obj][axis2]['composite_graph'])

                    if not composite_graph:
                        axis1compare = value_graphs[obj][axis1]['adjustment_curve']
                        axis2compare = value_graphs[obj][axis2]['adjustment_curve']
                        highest_intensity_curve = compare_curve_intensities(axis1compare, axis2compare)

                        if highest_intensity_curve == value_graphs[obj][axis1]['adjustment_curve']:
                            values = value_graphs[obj][axis1]['composite_graph']
                            if not is_equal(values): # Trying to escape flat composite graphs and encourage looking for other channels
                                print("substituting {0} for {1}".format(attr, axis1))
                                composite_graph = values
                        elif highest_intensity_curve == value_graphs[obj][axis2]['adjustment_curve']:
                            values = value_graphs[obj][axis2]['composite_graph']
                            if not is_equal(values): # Trying to escape flat composite graphs and encourage looking for other channels
                                print("substituting {0} for {1}".format(attr, axis2))
                                composite_graph = values
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
                new_value = 0.0

                for index, value in enumerate(frame_range):
                    sum_percentage += normalized_velocity_graph[index]
                    try:
                        new_value = map_from_to(sum_percentage, 0, 100, adjustment_graph[calculation_range.index(frange[0])], adjustment_graph[calculation_range.index(frange[1])])
                    except TypeError:
                        pass
                        # print("TypeError")
                        # if DEBUG:
                        #     print("frange = {}".format(frange))
                        #     print("sum_percentage = {}".format(sum_percentage))
                        #     print("adjustment_graph = {}".format(adjustment_graph))
                        #     print("adjustment_graph_frange0 = {}".format(adjustment_graph[calculation_range.index(frange[0])]))
                        #     print("adjustment_graph_frange1 = {}".format(adjustment_graph[calculation_range.index(frange[1])]))
                    if value not in frame_march:
                        new_value_curve.append(new_value)
                        frame_march.append(value) # I do this to skip the repeat frames between sets - those keys already exist anyway

            # Now set the keys
            # Do the magic, DO THE MAGIC!
            if do_set:
                # if DEBUG:
                #     print "Running adjustment on {}.".format(adjustment_curve)
                try:
                    for index, time in enumerate(adjustment_range):
                        cmds.setKeyframe(adjustment_curve, animLayer=adjustment_layer, time=(time,), value=new_value_curve[index])
                except:
                    if DEBUG:
                        print("adjustment_curve = {}".format(adjustment_curve))
                        print("adjustment_layer = {}".format(adjustment_layer))
                        print("new_value_curve[index] = {}".format(new_value_curve[index]))
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
    if not curve1: return curve2
    if not curve2: return curve1
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


# =============================================================================
# Developer Section
# =============================================================================

if __name__ == '__main__':
    print("# " + 76 * "=" + " #\n")  # Divider

    # Use the new pipeline architecture by default
    # Set USE_LEGACY=True to use the original implementation
    USE_LEGACY = False

    if USE_LEGACY:
        run(smart=False, do_set=True)
    else:
        run_pipeline(smart=False, do_set=True)


# =============================================================================
# TODO / Known Issues
# =============================================================================
#
# TODO: Evaluate sections independently from each other. For example, if an
#       adjustment goes from 1-90 and another from 90-100, if the baseAnimation
#       has no value change within the first section, but does within the second,
#       the composite is flat for the first section but not the second section.
#       Bad results.
#
# TODO: Support override layers properly (currently assumes additive)
#
