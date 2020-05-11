## AdjustmentBlend
Based on [Dan Low's GDC talk](https://www.youtube.com/watch?v=eeWBlMJHR14).

## Instructions:
Place adjustment_blend.py in your scripts folder.

1. Animate your objects. 
2. Add ONE animLayer to offset your animation. Key only the first and last frames on the animLayer.
3. Select all the objects that share that animLayer
4. Run the following command in Python:

##
    import adjustment_blend
    adjustment_blend.adjustment_blend_selected()

