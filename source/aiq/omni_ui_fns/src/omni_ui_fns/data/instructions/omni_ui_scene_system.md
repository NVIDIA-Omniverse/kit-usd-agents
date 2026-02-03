There are two main classes of omni.ui.scene Objects, the Shapes and the Containers.

# Shapes
Shape defines the 3D graphics-related item that is directly transformable. It is the base class for all "geometric primitives", which encodes several per-primitive graphics-related properties.
We have Line, Curve, Rectangle, Arc, Image, Points, PolygonMesh, TextureMesh and Label.

# Container
Container is the base class for grouping items. It's possible to add children to the container with Python's `with` statement. It's not possible to reparent items. Instead, it's necessary to remove the item and recreate a similar item under another parent.

## SceneView
`SceneView` is the `omni.ui` widget that renders all the SceneUI items. It can be a part of the `omni.ui` layout or an overlay of `omni.ui` interface. `SceneView().scene` is the entry point of SceneUI, not `Scene()`.

Never write something like this
```
from omni.ui import scene as sc

with sc.Scene():
    sc.Line([1, 0, 0], [0, 1, 0])
```

It should be
```
from omni.ui import scene as sc

scene_view = sc.SceneView(height=200)
with scene_view.scene:
    sc.Line([1, 0, 0], [0, 1, 0])
```

It is recommended to create a SceneView when writing omni.ui.scene code.


## Camera
SceneView determines the position and configuration of the camera and has
projection and view matrices.

```execute
from omni.ui import scene as sc

# Projection matrix
proj = [1.7, 0, 0, 0, 0, 3, 0, 0, 0, 0, -1, -1, 0, 0, -2, 0]

# Move camera
rotation = sc.Matrix44.get_rotation_matrix(30, 50, 0, True)
transl = sc.Matrix44.get_translation_matrix(0, 0, -6)
view = transl * rotation

scene_view = sc.SceneView(
    sc.CameraModel(proj, view),
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=200
)

with scene_view.scene:
    # Edges of cube
    sc.Line([-1, -1, -1], [1, -1, -1])
    sc.Line([-1, 1, -1], [1, 1, -1])
    sc.Line([-1, -1, 1], [1, -1, 1])
    sc.Line([-1, 1, 1], [1, 1, 1])

    sc.Line([-1, -1, -1], [-1, 1, -1])
    sc.Line([1, -1, -1], [1, 1, -1])
    sc.Line([-1, -1, 1], [-1, 1, 1])
    sc.Line([1, -1, 1], [1, 1, 1])

    sc.Line([-1, -1, -1], [-1, -1, 1])
    sc.Line([-1, 1, -1], [-1, 1, 1])
    sc.Line([1, -1, -1], [1, -1, 1])
    sc.Line([1, 1, -1], [1, 1, 1])
```

## Screen

To get position and view matrices, SceneView queries the model component. It
represents either the data transferred from the external backend or the data
that the model holds. The user can reimplement the model to manage the user
input or get the camera directly from the renderer or any other back end.

The SceneView model is required to return two float arrays, `projection`
and `view` of size 16, which are the camera matrices.

`ui.Screen` is designed to simplify tracking the user input to control the
camera position. `ui.Screen` represents the rectangle always placed in the front
of the camera. It doesn't produce any visible shape, but it interacts with the
user input.

You can change the camera orientation in the example below by dragging the mouse
cursor left and right.

```execute 200

from omni.ui import scene as sc

class CameraModel(sc.AbstractManipulatorModel):
    def __init__(self):
        super().__init__()
        self._angle = 0

    def append_angle(self, delta: float):
        self._angle += delta * 100
        # Inform SceneView that view matrix is changed
        self._item_changed("view")

    def get_as_floats(self, item):
        """Called by SceneView to get projection and view matrices"""
        if item == self.get_item("projection"):
            # Projection matrix
            return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, -1, 0, 0, -2, 0]
        if item == self.get_item("view"):
            # Move camera
            rotation = sc.Matrix44.get_rotation_matrix(30, self._angle, 0, True)
            transl = sc.Matrix44.get_translation_matrix(0, 0, -8)
            view = transl * rotation
            return [view[i] for i in range(16)]

def on_mouse_dragged(sender):
    # Change the model's angle according to mouse x offset
    mouse_moved = sender.gesture_payload.mouse_moved[0]
    sender.scene_view.model.append_angle(mouse_moved)

with sc.SceneView(CameraModel(), height=200).scene:
    # Camera control
    sc.Screen(gesture=sc.DragGesture(on_changed_fn=on_mouse_dragged))

    # Edges of cube
    sc.Line([-1, -1, -1], [1, -1, -1])
    sc.Line([-1, 1, -1], [1, 1, -1])
    sc.Line([-1, -1, 1], [1, -1, 1])
    sc.Line([-1, 1, 1], [1, 1, 1])

    sc.Line([-1, -1, -1], [-1, 1, -1])
    sc.Line([1, -1, -1], [1, 1, -1])
    sc.Line([-1, -1, 1], [-1, 1, 1])
    sc.Line([1, -1, 1], [1, 1, 1])

    sc.Line([-1, -1, -1], [-1, -1, 1])
    sc.Line([-1, 1, -1], [-1, 1, 1])
    sc.Line([1, -1, -1], [1, -1, 1])
    sc.Line([1, 1, -1], [1, 1, 1])
```

Here is a simple example of creating a red Line from coordinates (-0.5,-0.5,0) to (-0.5, 0.5, 0)

```
from omni.ui import scene as sc
from omni.ui import color as cl

scene_view = sc.SceneView(
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=200
)

with scene_view.scene:
    sc.Line([-0.5,-0.5,0], [-0.5, 0.5, 0], color=cl.red)
```

Transform is the container that propagates the affine transformations to its children. It has properties to scale the items to screen space and orient the
items to the current camera.

```execute 150
from omni.ui import scene as sc
from omni.ui import color as cl
import math

scene_view = sc.SceneView(
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=150,
)

with scene_view.scene:
    line_count = 36

    for i in range(line_count):
        weight = i / line_count
        angle = 2.0 * math.pi * weight

        # translation matrix
        move = sc.Matrix44.get_translation_matrix(
            8 * (weight - 0.5), 0.5 * math.sin(angle), 0)

        # rotation matrix
        rotate = sc.Matrix44.get_rotation_matrix(0, 0, angle)

        # the final transformation
        transform = move * rotate

        color = cl(weight, 1.0 - weight, 1.0)

        # create transform and put line to it
        with sc.Transform(transform=transform):
            sc.Line([0, 0, 0], [0.5, 0, 0], color=color)
```


You might sometimes need omni.ui element as well in the code e.g. omni.ui.Spacer() and omni.ui.Color etc. omni.ui is the python module which supports users to build customized widgets to create the user interface. omni.ui.scene depends on omni.ui.


## Declarative syntax

SceneUI uses declarative syntax, so it's possible to state what the manipulator
should do. For example, you can write that you want an item list consisting of
an image and lines. The code is simpler and easier to read. This declarative style applies to complex concepts like interaction with the
mouse pointer. A gesture can be easily added to almost any item with a few lines
of code. The system handles all of the steps needed to compute the intersection
with the mouse pointer and depth sorting if you click many items at runtime.
With this easy input, your manipulator comes ready very quickly.

## Manipulator

The manipulator is a very basic implementation of the slider in 3D space. The
main feature of the manipulator is that it redraws and recreates all the
children once the model is changed. It makes the code straightforward. It takes
the position and the slider value from the model, and when the user changes the
slider position, it processes a custom gesture. It doesn't write to the model
directly to let the user decide what to do with the new data and how the
manipulator should react to the modification. For example, if the user wants to
implement the snapping to the round value, it would be handy to do it in the
custom gesture.

## Model

The model contains the following named items:

 - `value` - the current value of the slider
 - `min` - the minimum value of the slider
 - `max` - the maximum value of the slider
 - `position` - the position of the slider in 3D space

The model demonstrates two main strategies working with the data.

The first strategy is that the model is the bridge between the manipulator and
the data, and it doesn't keep and doesn't duplicate the data. When the
manipulator requests the position from the model, the model computes the
position using USD API and returns it to the manipulator.

The first strategy is that the model can be a container of the data. For
example, the model pre-computes min and max values and passes them to the
manipulator once the selection is changed.

## Sync with USD Camera

To track the camera in the USD Stage, it's possible to use the standard USD way
to track the changes in UsdObjects which is a pert of Tf.Notice system.

`Usd.Notice.ObjectsChanged` sent in response to authored changes that affect
UsdObjects. The kinds of object changes are divided into two categories:
"resync" and "changed-info". We are interested in "changed-info" which means
that a nonstructural change has occurred, like an attribute value change or a
value change.

If the camera is in the changed-info list, we can extract view and projection
matrices and set them to the sc.SceneView object and it will redraw everything
with the new camera position and projection.

```python
import omni.usd

# Tracking the camera
stage = omni.usd.get_context().get_stage()
self._stage_listener = Tf.Notice.Register(
    Usd.Notice.ObjectsChanged, self._notice_changed, stage)

def _camera_changed(self):
    """Called when the camera is changed"""
    def flatten(transform):
        """Convert array[n][m] to array[n*m]"""
        return [item for sublist in transform for item in sublist]

    # Extract view and projection
    frustum = UsdGeom.Camera(self._camera_prim).GetCamera().frustum
    view = frustum.ComputeViewMatrix()
    projection = frustum.ComputeProjectionMatrix()

    # Convert Gf.Matrix4d to list
    view = flatten(view)
    projection = flatten(projection)

    # Set the scene
    self._scene_view.model.view = view
    self._scene_view.model.projection = projection

def _notice_changed(self, notice, stage):
    """Called by Tf.Notice"""
    for p in notice.GetChangedInfoOnlyPaths():
        if p.GetPrimPath() == self._camera_path:
            self._camera_changed()
```

## Standard Manipulators

The base manipulators for translate, rotate, and scale are provided in a
separate extension `omni.kit.manipulator.transform`. It's the same set of
manipulators that Kit is using. It's possible to enable the extension in the
Extensions explorer or click the button below.

```execute 30
import omni.kit.app

manager = omni.kit.app.get_app().get_extension_manager()

def enable(button, ext):
    manager.set_extension_enabled_immediate(ext, True)
    button.text = "Extension is Enabled"

if manager.is_extension_enabled("omni.kit.manipulator.transform"):
    label_enabled = "Extension is Enabled"
else:
    label_enabled = "Click to Enable Extension omni.kit.manipulator.transform"

button = ui.Button(label_enabled, height=30)
button.set_clicked_fn(lambda: enable(button, "omni.kit.manipulator.transform"))
```

TransformManipulator is a model-based manipulator. It can be used to scale,
translate or rotate custom items like objects, control points, or helpers in
both 2d and 3d views.

Like all the model-based items, to use it, it's necessary to either reimplement
the model or subscribe to the model's changes.

The reimplemented model should have four mandatory fields. Listening to the
following fields is necessary when subscribing to the model changes.

 - `transform`: is used to set the position of the manipulator
 - `translate`: the manipulator sets this value when it's being dragged
 - `rotate`: the manipulator sets this value when it's being rotated
 - `scale`: the manipulator sets this value when it's being scaled

It's the same set of manipulators that Kit is using. The advantage of using it
is that it's always up to date, and the look is consistent with the Viewport
manipulator.

In this simple example, it's possible to select and move points with the mouse.

```execute 200
import omni.kit.app
from omni.ui import scene as sc

self.frame = ui.Frame(height=200)

self.manager = omni.kit.app.get_app().get_extension_manager()
self.hook = self.manager.subscribe_to_extension_enable(
    lambda _: self.frame.rebuild(),
    lambda _: self.frame.rebuild(),
    ext_name="omni.kit.manipulator.transform",
    hook_name="omni.ui.scene.docs listener",
)


def build():
    if not self.manager.is_extension_enabled("omni.kit.manipulator.transform"):
        ui.Spacer()
        return

    from omni.kit.manipulator.transform.manipulator import TransformManipulator
    from omni.kit.manipulator.transform.manipulator import Axis
    from omni.kit.manipulator.transform.simple_transform_model import SimpleRotateChangedGesture
    from omni.kit.manipulator.transform.simple_transform_model import SimpleScaleChangedGesture
    from omni.kit.manipulator.transform.simple_transform_model import SimpleTranslateChangedGesture
    from omni.kit.manipulator.transform.types import Operation

    # Camera matrices
    projection = [1e-2, 0, 0, 0]
    projection += [0, 1e-2, 0, 0]
    projection += [0, 0, -2e-7, 0]
    projection += [0, 0, 1, 1]
    view = sc.Matrix44.get_translation_matrix(0, 0, -5)

    # Selected point
    self._selected_item = None

    def _on_point_clicked(shape):
        """Called when the user clicks the point"""
        self._selected_item = shape

        pos = self._selected_item.positions[0]
        model = self._manipulator.model
        model.set_floats(model.get_item("translate"), [pos[0], pos[1], pos[2]])

    def _on_item_changed(model, item):
        """
        Called when the user moves the manipulator. We need to move
        the point here.
        """
        if self._selected_item is not None:
            if item.operation == Operation.TRANSLATE:
                self._selected_item.positions = model.get_as_floats(item)

    with sc.SceneView(sc.CameraModel(projection, view)).scene:
        # The manipulator
        self._manipulator = TransformManipulator(
            size=1,
            axes=Axis.ALL & ~Axis.Z & ~Axis.SCREEN,
            gestures=[
                SimpleTranslateChangedGesture(),
                SimpleRotateChangedGesture(),
                SimpleScaleChangedGesture(),
            ],
        )

        self._sub = \
            self._manipulator.model.subscribe_item_changed_fn(_on_item_changed)

        # 5 points
        select = sc.ClickGesture(_on_point_clicked)
        sc.Points([[-5, 5, 0]], colors=[ui.color.white], sizes=[10], gesture=select)
        sc.Points([[5, 5, 0]], colors=[ui.color.white], sizes=[10], gesture=select)
        sc.Points([[5, -5, 0]], colors=[ui.color.white], sizes=[10], gesture=select)
        sc.Points([[-5, -5, 0]], colors=[ui.color.white], sizes=[10], gesture=select)
        self._selected_item = sc.Points(
            [[0, 0, 0]], colors=[ui.color.white], sizes=[10], gesture=select
        )

self.frame.set_build_fn(build)

```
# Gestures.md

# Gestures

![](gestures.png)

Gestures handle all of the logic needed to process user-input events such as
click and drag and recognize when those events happen on the shape. When
recognizing it, SceneUI runs a callback to update the state of a view or perform
an action.

## Add Gesture Callback

Each gesture applies to a specific shape in the scene hierarchy. To recognize a
gesture event on a particular shape, it's necessary to create and configure the
gesture object. SceneUI provides the full state of the gesture to the callback
using the `gesture_payload` object. The `gesture_payload` object contains the
coordinates of the itersection of mouse pointer and the shape in world and shape
spaces, the distance between last itersection and parametric coordinates.
It's effortless to modify the state of the shape using this data.

```execute 200
from omni.ui import scene as sc
from omni.ui import color as cl
from functools import partial

proj = [0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 2e-7, 0, 0, 0, 1, 1]
view = sc.Matrix44.get_translation_matrix(0, 0, -10)
scene_view = sc.SceneView(
    sc.CameraModel(proj, view),
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=200
)

def move(transform: sc.Transform, shape: sc.AbstractShape):
    """Called by the gesture"""
    translate = shape.gesture_payload.moved
    # Move transform to the direction mouse moved
    current = sc.Matrix44.get_translation_matrix(*translate)
    transform.transform *= current

with scene_view.scene:
    transform = sc.Transform()
    with transform:
        sc.Line(
            [-1, 0, 0],
            [1, 0, 0],
            color=cl.blue,
            thickness=5,
            gesture=sc.DragGesture(
                on_changed_fn=partial(move, transform)
            )
        )
```

## Reimplementing Gesture

Some gestures can receive the update in a different state. For example,
DragGesture offers the update when the user presses the mouse, moves the mouse,
and releases the mouse: `on_began`, `on_changed`, `on_ended`. It's also possible
to extend the gesture by reimplementing its class.

```execute 200
##
from omni.ui import scene as sc
from omni.ui import color as cl
from functools import partial

proj = [0.3, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 2e-7, 0, 0, 0, 1, 1]
view = sc.Matrix44.get_translation_matrix(0, 0, -10)
scene_view = sc.SceneView(
    sc.CameraModel(proj, view),
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=200
)

class Move(sc.DragGesture):
    def __init__(self, transform: sc.Transform):
        super().__init__()
        self.__transform = transform

    def on_began(self):
        self.sender.color = cl.red

    def on_changed(self):
        translate = self.sender.gesture_payload.moved
        # Move transform to the direction mouse moved
        current = sc.Matrix44.get_translation_matrix(*translate)
        self.__transform.transform *= current

    def on_ended(self):
        self.sender.color = cl.blue

with scene_view.scene:
    transform = sc.Transform()
    with transform:
        sc.Rectangle(color=cl.blue, gesture=Move(transform))
```

## Gesture Manager

Gestures track incoming input events separately, but it's normally necessary to
let only one gesture be executed because it prevents user input from triggering
more than one action at a time. For example, if multiple shapes are under the
mouse pointer, normally, the only gesture of the closest shape should be
processed. However, this default behavior can introduce unintended side effects.
To solve those problems, it's necessary to use the gesture manager.

GestureManager controls the priority of gestures if they are processed at the
same time. It prioritizes the desired gestures and prevents unintended gestures
from being executed.

In the following example, the gesture of the red rectangle always wins even if
both rectangles overlap.

```execute 200
from omni.ui import scene as sc
from omni.ui import color as cl
from functools import partial

proj = [0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 2e-7, 0, 0, 0, 1, 1]
view = sc.Matrix44.get_translation_matrix(0, 0, -10)
scene_view = sc.SceneView(
    sc.CameraModel(proj, view),
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=200
)

class Manager(sc.GestureManager):
    def __init__(self):
        super().__init__()

    def should_prevent(self, gesture, preventer):
        # prime gesture always wins
        if preventer.name == "prime":
            return True

def move(transform: sc.Transform, shape: sc.AbstractShape):
    """Called by the gesture"""
    translate = shape.gesture_payload.moved
    current = sc.Matrix44.get_translation_matrix(*translate)
    transform.transform *= current

mgr = Manager()

with scene_view.scene:
    transform1 = sc.Transform()
    with transform1:
        sc.Rectangle(
            color=cl.red,
            gesture=sc.DragGesture(
                name="prime",
                manager=mgr,
                on_changed_fn=partial(move, transform1)
            )
        )
    transform2 = sc.Transform(
        transform=sc.Matrix44.get_translation_matrix(2, 0, 0)
    )
    with transform2:
        sc.Rectangle(
            color=cl.blue,
            gesture=sc.DragGesture(
                manager=mgr,
                on_changed_fn=partial(move, transform2)
            )
        )
```

# Manipulator

![](manipulator.png)

Manipulator provides a model-based template that is flexible for implementing
navigation and editing objects, points, and properties. It's a container that
can be reimplemented. It can have a model.

## Immediate mode

It's possible to regenerate the content of the Manipulator item by calling
the `invalidate` method. Once it's called, Manipulator will flush the old
children and execute `build_fn` to create new ones. Suppose the `invalidate`
method is called inside `build_fn`. In that case, Manipulator will
call `build_fn` every frame, which provides, on the one hand, a maximum of
control and flexibility to the scene configuration. Still, on the other hand,
it also generates a continuous workload on the CPU.

```execute 200
from omni.ui import scene as sc
from omni.ui import color as cl

class RotatingCube(sc.Manipulator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._angle = 0

    def on_build(self):
        transform = sc.Matrix44.get_rotation_matrix(
            0, self._angle, 0, True)

        with sc.Transform(transform=transform):
            sc.Line([-1, -1, -1], [1, -1, -1])
            sc.Line([-1, 1, -1], [1, 1, -1])
            sc.Line([-1, -1, 1], [1, -1, 1])
            sc.Line([-1, 1, 1], [1, 1, 1])

            sc.Line([-1, -1, -1], [-1, 1, -1])
            sc.Line([1, -1, -1], [1, 1, -1])
            sc.Line([-1, -1, 1], [-1, 1, 1])
            sc.Line([1, -1, 1], [1, 1, 1])

            sc.Line([-1, -1, -1], [-1, -1, 1])
            sc.Line([-1, 1, -1], [-1, 1, 1])
            sc.Line([1, -1, -1], [1, -1, 1])
            sc.Line([1, 1, -1], [1, 1, 1])

        # Increase the angle
        self._angle += 1

        # Redraw all
        self.invalidate()

# Projection matrix
proj = [1.7, 0, 0, 0, 0, 3, 0, 0, 0, 0, -1, -1, 0, 0, -2, 0]

# Move camera
rotation = sc.Matrix44.get_rotation_matrix(30, 50, 0, True)
transl = sc.Matrix44.get_translation_matrix(0, 0, -6)
view = transl * rotation

scene_view = sc.SceneView(
    sc.CameraModel(proj, view),
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=200
)

with scene_view.scene:
    RotatingCube()
```

## Model

The model is the central component of Manipulator. It is the dynamic data
structure, independent of the user interface. It can contain nested data, but
it's supposed to be the bridge between the user data and the Manipulator object.
It follows the closely model-view pattern.

When the model is changed, it calls `on_model_updated` of Manipulator. Thus,
Manipulator can modify the children or rebuild everything completely depending
on the change.

It's abstract, and it is not supposed to be instantiated directly. Instead, the
user should subclass it to create the new model.

```execute 200
from omni.ui import scene as sc
from omni.ui import color as cl

class MovingRectangle(sc.Manipulator):
    """Manipulator that redraws when the model is changed"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._gesture = sc.DragGesture(on_changed_fn=self._move)

    def on_build(self):
        position = self.model.get_as_floats(self.model.get_item("position"))
        transform = sc.Matrix44.get_translation_matrix(*position)
        with sc.Transform(transform=transform):
            sc.Rectangle(color=cl.blue, gesture=self._gesture)

    def on_model_updated(self, item):
        self.invalidate()

    def _move(self, shape: sc.AbstractShape):
        position = shape.gesture_payload.ray_closest_point
        item = self.model.get_item("position")
        self.model.set_floats(item, position)


class Model(sc.AbstractManipulatorModel):
    """User part. Simple value holder."""

    class PositionItem(sc.AbstractManipulatorItem):
        def __init__(self):
            super().__init__()
            self.value = [0, 0, 0]

    def __init__(self):
        super().__init__()
        self.position = Model.PositionItem()

    def get_item(self, identifier):
        return self.position

    def get_as_floats(self, item):
        return item.value

    def set_floats(self, item, value):
        item.value = value
        self._item_changed(item)


# Projection matrix
proj = [1.7, 0, 0, 0, 0, 3, 0, 0, 0, 0, -1, -1, 0, 0, -2, 0]
# Move/rotate camera
rotation = sc.Matrix44.get_rotation_matrix(30, 30, 0, True)
transl = sc.Matrix44.get_translation_matrix(0, 0, -6)

scene_view = sc.SceneView(
    sc.CameraModel(proj, transl * rotation),
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=200
)

with scene_view.scene:
    MovingRectangle(model=Model())
```
