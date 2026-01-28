---
title: How to create 3D graphics using DirectX in MetaTrader 5
url: https://www.mql5.com/en/articles/7708
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:14:37.541002
---

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/7708&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071685064946822283)

MetaTrader 5 / Integration


Three-dimensional computer graphics provide the impressions of three-dimensional objects on a flat display. Such objects, as well as the position of the
viewer can change over time. Accordingly, the two-dimensional picture should also change to create the illusion of the image depth, i.e. it
should support rotation, zooming, changes in lighting and so on. MQL5 allows creating and managing computer graphics directly in the
MetaTrader 5 terminal using [DirectX functions](https://www.mql5.com/en/docs/directx). Please note that your video card
should support [DX \\
11](https://en.wikipedia.org/wiki/DirectX#DirectX_11 "https://en.wikipedia.org/wiki/DirectX#DirectX_11") and Shader Model 5.0 for the functions to work.

- [Object Modeling](https://www.mql5.com/en/articles/7708#model)

- [Creating a Shape](https://www.mql5.com/en/articles/7708#create)
- [Scene Calculation and Rendering](https://www.mql5.com/en/articles/7708#render)

- [Object Rotation around the Z Axis and Viewpoint](https://www.mql5.com/en/articles/7708#rotation)
- [Camera Position Management](https://www.mql5.com/en/articles/7708#camera_position)
- [Object Color Management](https://www.mql5.com/en/articles/7708#object_color)
- [Rotation and Movement](https://www.mql5.com/en/articles/7708#transformation)
- [Working with Lighting](https://www.mql5.com/en/articles/7708#scene_light)
- [Animation](https://www.mql5.com/en/articles/7708#animation)
- [Control Camera Position Using the Mouse](https://www.mql5.com/en/articles/7708#camera_by_mouse)
- [Applying Textures](https://www.mql5.com/en/articles/7708#texture)
- [Creating Custom Objects](https://www.mql5.com/en/articles/7708#custom_object)
- [Data-Based 3D Surface](https://www.mql5.com/en/articles/7708#3d_surface)

### Object Modeling

To draw a three-dimensional object on a flat space, a model of this object in X, Y and Z coordinates should be obtained first. It means that
each point on the object surface should be described by specifying its coordinates. Ideally, one would need to describe an infinite
number of points on the object surface to preserve the image quality during scaling. In practice, 3D models are described using a mesh
consisting of polygons. A more detailed mesh with a higher number of polygons provides a more realistic model. However, more computer
resources are required to calculate such a model and to render 3D graphics.

![A teapot model as a polygon mesh](https://c.mql5.com/2/38/kettle.png)

A teapot model as a polygon mesh.

The division of polygons into triangles appeared long ago when early computer graphics had to run on weak graphics cards. The triangle
enables the exact description of the position of a small surface part, as well as the calculation of related parameters, such as lights
and light reflections. The collection of such small triangles allows creating a realistic three-dimensional image of the object.
Hereinafter, the polygon and the triangle will be used as synonyms since it is much easier to imagine a triangle than a polygon with N
vertices.

![](https://c.mql5.com/2/38/cube.png)

Cube made up of triangles.

A three-dimensional model of an object can be created by describing the coordinates of each vertex of the triangle, which allows
further calculation of coordinates for each point of the object, even if the object moves or the viewer's position changes. Thus, we
deal with vertices, the edges that connect them, and the face which is formed by the edges. If the position of a triangle is known, we can
create a normal for the face using the laws of linear algebra (a normal is a vector that is perpendicular to the surface). This allows
calculating how the face will be lighted and how the light will be reflected from it.

[https://c.mql5.com/2/38/2020-04-06_20h56_38.png](https://c.mql5.com/2/38/2020-04-06_20h56_38.png "https://c.mql5.com/2/38/2020-04-06_20h56_38.png")[![](https://c.mql5.com/2/38/elements.png)](https://c.mql5.com/2/38/ckyjxjxqzed.png "https://c.mql5.com/2/38/ckyjxjxqzed.png")

Examples of simple objects with vertices, edges, faces and normals. A normal is a red
arrow.

A model object can be created in different ways. Topology describes how polygons form the 3D mesh. A good topology allows using the
minimum number of polygons to describe an object and can make it easier to move and rotate the object.

![Sphere model in two topologies](https://c.mql5.com/2/38/topology.png)

Sphere model in two topologies.

The volume effect is created by using lights and shadows on the object polygons. Thus, the purpose of 3D computer graphics is to calculate
the position of each point of an object, to calculate lights and shadows and to display it on the screen.

### Creating a Shape

Let us write a simple program that creates a cube. Use the [CCanvas3D](https://www.mql5.com/en/docs/standardlibrary/3dgraphics/ccanvas3d)
class from the [3D graphics](https://www.mql5.com/en/docs/standardlibrary/3dgraphics "3D graphics") library.

The CCanvas3DWindow class, which renders a 3D window, has a minimum of members and methods. We will gradually add new methods with an
explanation of 3D graphics concepts implemented in [functions for working with DirectX](https://www.mql5.com/en/docs/directx).

```
//+------------------------------------------------------------------+
//| Application window                                               |
//+------------------------------------------------------------------+
class CCanvas3DWindow
  {
protected:
   CCanvas3D         m_canvas;
   //--- canvas size
   int               m_width;
   int               m_height;
   //--- the Cube object
   CDXBox            m_box;

public:
                     CCanvas3DWindow(void) {}
                    ~CCanvas3DWindow(void) {m_box.Shutdown();}
   //-- create a scene
   virtual bool      Create(const int width,const int height){}
   //--- calculate the scene
   void              Redraw(){}
   //--- handle chart events
   void              OnChartChange(void) {}
  };
```

Creation of a scene starts with the creation of a canvas. Then the following parameters are set for the projection matrix:

1. A 30-degree angle of view (M\_PI/6), from which we look at the 3D scene
2. Aspect ratio as a ratio of width and height
3. Distance to the near (0.1f) and far (100.f) clipping plane

This means that only the objects between these two virtual walls (0.1f and 100.f) will be rendered in the projection matrix. In addition, the
object must fall into the horizontal 30-degree angle of view. Please note that distances as well as all the coordinates in computer graphics
are virtual. What matters is the relationships between the distances and sizes, but not the absolute values.

```
   //+------------------------------------------------------------------+
   //| Create                                                           |
   //+------------------------------------------------------------------+
   virtual bool      Create(const int width,const int height)
     {
      //--- save canvas dimensions
      m_width=width;
      m_height=height;
      //--- create a canvas to render a 3D scene
      ResetLastError();
      if(!m_canvas.CreateBitmapLabel("3D Sample_1",0,0,m_width,m_height,COLOR_FORMAT_ARGB_NORMALIZE))
        {
         Print("Error creating canvas: ",GetLastError());
         return(false);
         }
      //--- set projection matrix parameters - angle of view, aspect ratio, distance to the near and far clip planes
      m_canvas.ProjectionMatrixSet((float)M_PI/6,(float)m_width/m_height,0.1f,100.0f);
      //--- create cube - pass to it the resource manager, scene parameters and coordinates of two opposite corners of the cube
      if(!m_box.Create(m_canvas.DXDispatcher(),m_canvas.InputScene(),DXVector3(-1.0,-1.0,5.0),DXVector3(1.0,1.0,7.0)))
        {
         m_canvas.Destroy();
         return(false);
         }
      //--- add the cube to the scene
      m_canvas.ObjectAdd(&m_box);
      //--- redraw the scene
      Redraw();
      //--- succeed
      return(true);
      }
```

After creating the projection matrix, we can proceed to constructing the 3D object — a cube based on the CDXBox class. To create a cube, it is
enough to indicate two vectors pointing to the opposite corners of the cube. By watching the cube creation in the debug mode, you can see what
is happening in DXComputeBox(): the creation of all the cube vertices (their coordinates are written to the 'vertices' array), as well as
dividing of cube edges into triangles which are enumerated and saved in the 'indiсes' array. In total, the cube has 8 vertices, 6 faces which
are divided into 12 triangles, and 36 indices enumerating the vertices of these triangles.

Although the cube has only 8 vertices, 24 vectors are created to describe them, since a separate set of vertices having a normal should be specified
for each of the 6 faces. The direction of the normal will affect the calculation of the lighting for each face. The order in which the vertices
of a triangle are listed in the index determines which of its sides will be visible. The order in which the vertices and indices are filled is
shown in the DXUtils.mqh code:

```
   for(int i=20; i<24; i++)
      vertices[i].normal=DXVector4(0.0,-1.0,0.0,0.0);
```

Texture coordinates for texture mapping for each face are described in the same code:

```
//--- texture coordinates
   for(int i=0; i<faces; i++)
     {
      vertices[i*4+0].tcoord=DXVector2(0.0f,0.0f);
      vertices[i*4+1].tcoord=DXVector2(1.0f,0.0f);
      vertices[i*4+2].tcoord=DXVector2(1.0f,1.0f);
      vertices[i*4+3].tcoord=DXVector2(0.0f,1.0f);
      }
```

Each of the 4 face vectors sets one of the 4 angles for texture mapping. This means that a squad structure will be mapped to each cube face to render
the texture. Of course, this is required only if the texture is set.

### Scene Calculation and Rendering

All calculations should be performed anew every time the 3D scene is changed. Here is the order of required calculations:

- Calculate the center of each object in world coordinates
- Calculate the position of each element of the object, i.e. of each vertex
- Determine pixel depth and its visibility for the viewer

- Calculate the position of each pixel on the polygon specified by its vertices
- Set the color of each pixel on the polygon in accordance with the specified texture

- Calculate the direction of the light pixel and its reflection
- Apply diffused light to each pixel
- Convert all world coordinates into camera coordinates
- Convert camera coordinates into the coordinates on the projection matrix

All these operations are performed in the Render method of the CCanvas3D object. After rendering, the calculated image is transferred from
the projection matrix to the canvas by calling the Update method.

```
   //+------------------------------------------------------------------+
   //| Update the scene                                                 |
   //+------------------------------------------------------------------+
   void              Redraw()
     {
      //--- calculate the 3D scene
      m_canvas.Render(DX_CLEAR_COLOR|DX_CLEAR_DEPTH,ColorToARGB(clrBlack));
      //--- update the picture on the canvas in accordance with the current scene
      m_canvas.Update();
      }
```

In our example, the cube is created only once, and it does not change any more. Therefore, the frame on the canvas will only need to be changed if
there are changes in the chart, such as the chart resizing. In this case, the canvas dimensions are adjusted to the current chart dimensions,
the projection matrix is reset and an image on the canvas is updated.

```
   //+------------------------------------------------------------------+
   //| Process chart change event                                       |
   //+------------------------------------------------------------------+
   void              OnChartChange(void)
     {
      //--- get current chart sizes
      int w=(int)ChartGetInteger(0,CHART_WIDTH_IN_PIXELS);
      int h=(int)ChartGetInteger(0,CHART_HEIGHT_IN_PIXELS);
      //--- update canvas dimensions in accordance with the chart size
      if(w!=m_width || h!=m_height)
        {
         m_width =w;
         m_height=h;
         //--- resize canvas
         m_canvas.Resize(w,h);
         DXContextSetSize(m_canvas.DXContext(),w,h);
         //--- update projection matrix in accordance with the canvas sizes
         m_canvas.ProjectionMatrixSet((float)M_PI/6,(float)m_width/m_height,0.1f,100.0f);
         //--- recalculate 3D scene and render it onto the canvas
         Redraw();
         }
      }
```

Launch the _"Step1 Create Box.mq5_" EA. You will see a white square on a black background. By default, white color is set for
objects upon creation. Lighting has not yet been set.

![A white cube and its layout in space](https://c.mql5.com/2/38/box_in_3d_scene__2.png)

A white cube and its layout in space

The X axis is directed to the right, Y is directed upward, and Z is directed inward into the 3D scene. Such a coordinate system is called
left-handed.

The center of the cube is at the point with the following coordinates X=0, Y=0, Z=6. The position from which we look at the cube is in the center of
coordinates, which is the default value. If you want to change the position from which the 3D scene is viewed, explicitly set appropriate
coordinates using the [ViewPositionSet()](https://www.mql5.com/en/docs/standardlibrary/3dgraphics/ccanvas3d/ccanvas3dviewpositionset)
function.

To complete program operation, press "Escape".

### Object Rotation around the Z Axis and Viewpoint

To animate the scene, let us enable cube rotation around the Z axis. To do this, add a timer — based on its events the cube will be rotated
counterclockwise.

Create a rotation matrix to enable the rotation around the Z axis at a given angle using the DXMatrixRotationZ() method. Then pass it as a
parameter to the TransformMatrixSet() method. This will change the position of the cube in the 3D space. Again, call Redraw() to update the
image on the canvas.

```
   //+------------------------------------------------------------------+
   //| Timer handler                                                    |
   //+------------------------------------------------------------------+
   void              OnTimer(void)
     {
      //--- variables for calculating the rotation angle
      static ulong last_time=0;
      static float angle=0;
      //--- get the current time
      ulong current_time=GetMicrosecondCount();
      //--- calculate the delta
      float deltatime=(current_time-last_time)/1000000.0f;
      if(deltatime>0.1f)
         deltatime=0.1f;
      //--- increase the angle of rotation of the cube around the Z axis
      angle+=deltatime;
      //--- remember the time
      last_time=current_time;
      //--- set the angle of rotation of the cube around the Z axis
      DXMatrix rotation;
      DXMatrixRotationZ(rotation,angle);
      m_box.TransformMatrixSet(rotation);
      //--- recalculate 3D scene and render it onto the canvas
      Redraw();
      }
```

After launch, you will see a rotating white square.

![](https://c.mql5.com/2/38/box_rotation_z__4.gif)

The cube rotates around the Z axis counterclockwise

The source code of this example is available in the file _"Step2 Rotation Z.mq5_". Please note that angle **M\_PI/5**
is specified now when creating the scene, which is greater than angle M\_PI/6 from the previous example.

```
      //--- set projection matrix parameters - angle of view, aspect ratio, distance to the near and far clip planes
      m_matrix_view_angle=(float)M_PI/5;
      m_canvas.ProjectionMatrixSet(m_matrix_view_angle,(float)m_width/m_height,0.1f,100.0f);
      //--- create cube - pass to it the resource manager, scene parameters and coordinates of two opposite corners of the cube
```

However, the cube dimension in the screen are visually smaller. The smaller the angle of view specified when setting the projection matrix, the
larger frame part is occupied by the object. This can be compared to seeing objects with a telescope: the object is larger, though the angle of
view is smaller.

### Camera Position Management

The CCanvas3D class has three methods for setting important 3D scene parameters, which are interconnected:

- [ViewPositionSet](https://www.mql5.com/en/docs/standardlibrary/3dgraphics/ccanvas3d/ccanvas3dviewpositionset "ViewPositionSet")
sets the viewpoint of the 3D scene

- [ViewTargetSet](https://www.mql5.com/en/docs/standardlibrary/3dgraphics/ccanvas3d/ccanvas3dviewtargetset "ViewTargetSet")
sets the coordinates of the gaze point

- [ViewUpDirectionSet](https://www.mql5.com/en/docs/standardlibrary/3dgraphics/ccanvas3d/ccanvas3dviewupdirectionset "ViewUpDirectionSet")
sets the direction of the upper border of the frame in the 3D space


All these parameters are used in combination — this means that if you want to set any of these parameters in the 3D scene, the other two
parameters must also be initialized. This should be done at least at the scene generation stage. This is shown in the following example, in
which the upper border of the frame swings left and right. The swing is implemented by adding the following three code lines in the Create()
method:

```
   //+------------------------------------------------------------------+
   //| Create                                                           |
   //+------------------------------------------------------------------+
   virtual bool      Create(const int width,const int height)
     {
....
      //--- add the cube to the scene
      m_canvas.ObjectAdd(&m_box);
      //--- set the scene parameters
      m_canvas.ViewUpDirectionSet(DXVector3(0,1,0));  // set the direction vector up, along the Y axis
      m_canvas.ViewPositionSet(DXVector3(0,0,0));     // set the viewpoint from the center of coordinates
      m_canvas.ViewTargetSet(DXVector3(0,0,6));       // set the gaze point at center of the cube
      //--- redraw the scene
      Redraw();
      //--- succeed
      return(true);
      }
```

Modify the OnTimer() method to make the horizon vector swing left and right.

```
   //+------------------------------------------------------------------+
   //| Timer handler                                                    |
   //+------------------------------------------------------------------+
   void              OnTimer(void)
     {
      //--- variables for calculating the rotation angle
      static ulong last_time=0;
      static float max_angle=(float)M_PI/30;
      static float time=0;
      //--- get the current time
      ulong current_time=GetMicrosecondCount();
      //--- calculate the delta
      float deltatime=(current_time-last_time)/1000000.0f;
      if(deltatime>0.1f)
         deltatime=0.1f;
      //--- increase the angle of rotation of the cube around the Z axis
      time+=deltatime;
      //--- remember the time
      last_time=current_time;
      //--- set the rotation angle around the Z axis
      DXVector3 direction=DXVector3(0,1,0);     // initial direction of the top
      DXMatrix rotation;                        // rotation vector
      //--- calculate the rotation matrix
      DXMatrixRotationZ(rotation,float(MathSin(time)*max_angle));
      DXVec3TransformCoord(direction,direction,rotation);
      m_canvas.ViewUpDirectionSet(direction);   // set the new direction of the top
      //--- recalculate 3D scene and render it onto the canvas
      Redraw();
      }
```

Save the example as _"Step3 ViewUpDirectionSet.mq5"_ and run it. You will see the image of a swinging cube, although it is
actually motionless. This effect is obtained when the camera itself swings left and right.

![Top direction of the top swings left and right](https://c.mql5.com/2/38/box_pendal__2.gif)

Top direction of the top swings left and right

Remembered that there is a connection between the coordinates of the target, of the camera and of the direction of the top. Thus, in order to control
the position of the camera, you must also specify the direction of the top and the coordinates of the target, i.e. the gaze point.

### Object Color Management

Let us modify our code and put the cube in the center of coordinate, while moving the camera.

```
   //+------------------------------------------------------------------+
   //| Create                                                           |
   //+------------------------------------------------------------------+
   virtual bool      Create(const int width,const int height)
     {
  ...
      //--- create cube - pass to it the resource manager, scene parameters and coordinates of two opposite corners of the cube
      if(!m_box.Create(m_canvas.DXDispatcher(),m_canvas.InputScene(),DXVector3(-1.0,-1.0,-1.0),DXVector3(1.0,1.0,1.0)))
        {
         m_canvas.Destroy();
         return(false);
         }
      //--- set the color
      m_box.DiffuseColorSet(DXColor(0.0,0.5,1.0,1.0));
      //--- add the cube to the scene
      m_canvas.ObjectAdd(&m_box);
      //--- set positions for camera, gaze and direction of the top
      m_canvas.ViewUpDirectionSet(DXVector3(0.0,1.0,0.0));  // set the direction vector up, along the Y axis
      m_canvas.ViewPositionSet(DXVector3(3.0,2.0,-5.0));    // set camera on the right, on top and in front of the cube
      m_canvas.ViewTargetSet(DXVector3(0,0,0));             // set the gaze direction at center of the cube
      //--- redraw the scene
      Redraw();
      //--- succeed
      return(true);
      }
```

In addition, paint the cube in blue. The color is set in the format of an [RGB \\
color with an alpha channel](https://www.mql5.com/en/docs/convert/colortoargb) (the alpha channel is indicated last), though the values are normalized to one. Thus, a value of 1 means 255,
and 0.5 means 127.

Add rotation around the X axis and save changes as _"Step4 Box Color.mq5"_.

![Top right view of a rotating cube.](https://c.mql5.com/2/38/blue_rotation_x.gif)

Top right view of a rotating cube.

### Rotation and Movement

Objects can be moved and rotated in three directions at a time. All object changes are implemented using matrices. Each of them, i.e. rotation,
movement and transformation, can be calculated separately. Let us change the example: the camera view is now from top and front.

```
   //+------------------------------------------------------------------+
   //| Create                                                           |
   //+------------------------------------------------------------------+
   virtual bool      Create(const int width,const int height)
     {
  ...
      m_canvas.ProjectionMatrixSet(m_matrix_view_angle,(float)m_width/m_height,0.1f,100.0f);
      //--- position the camera in top and in front of the center of coordinates
      m_canvas.ViewPositionSet(DXVector3(0.0,2.0,-5.0));
      m_canvas.ViewTargetSet(DXVector3(0.0,0.0,0.0));
      m_canvas.ViewUpDirectionSet(DXVector3(0.0,1.0,0.0));
      //--- create cube - pass to it the resource manager, scene parameters and coordinates of two opposite corners of the cube
      if(!m_box.Create(m_canvas.DXDispatcher(),m_canvas.InputScene(),DXVector3(-1.0,-1.0,-1.0),DXVector3(1.0,1.0,1.0)))
        {
         m_canvas.Destroy();
         return(false);
         }
      //--- set the cube color
      m_box.DiffuseColorSet(DXColor(0.0,0.5,1.0,1.0));
      //--- calculate the cube position and the transfer matrix
      DXMatrix rotation,translation;
      //--- rotate the cube sequentially along the X, Y and Z axes
      DXMatrixRotationYawPitchRoll(rotation,(float)M_PI/4,(float)M_PI/3,(float)M_PI/6);
      //-- move the cube to the right/downward/inward
      DXMatrixTranslation(translation,1.0,-2.0,5.0);
      //--- get the transformation matrix as a product of rotation and transfer
      DXMatrix transform;
      DXMatrixMultiply(transform,rotation,translation);
      //--- set the transformation matrix
      m_box.TransformMatrixSet(transform);
      //--- add the cube to the scene
      m_canvas.ObjectAdd(&m_box);
      //--- redraw the scene
      Redraw();
      //--- succeed
      return(true);
      }
```

Sequentially create rotation and transfer matrices, apply the resulting transformation matrix and render the cube. Save changes in _"Step5_
_Translation.mq_ 5" and run it.

![Rotation and movement of a cube](https://c.mql5.com/2/38/box_translation.png)

Rotation and movement of a cube

The camera is still, and it is pointed to the center of coordinates a little from above. The cube was rotated in
three directions and was shifted to the right, down and inwards into the scene.

### Working with Lighting

To obtain a realistic three-dimensional image, it is necessary to calculate lighting of each point on the object surface. This is done using
the [Phong \\
shading model](https://en.wikipedia.org/wiki/Phong_shading "https://en.wikipedia.org/wiki/Phong_shading"), which calculates the color intensity of the following three lighting components: ambient, diffuse and specular. The
following parameters are used here:

- DirectionLight — the direction of the directional lighting is set in CCanvas3D

- AmbientLight — the color and intensity of the ambient lighting is set in CCanvas3D

- DiffuseColor — the calculated diffused lighting component is set in CDXMesh and its child classes

- EmissionColor — the background lighting component is set in CDXMesh and its child classes

- SpecularColor — the specular component is set in CDXMesh and its child classes

![Phong shading model](https://c.mql5.com/2/38/phong_model.png)

Phong shading model

The lighting model is implemented in standard shaders, the model parameters are set in CCanvas3D, and object parameters are set in CDXMesh
and its child classes. Modify the example as follows:

1. Return the cube to the center of coordinates.
2. Set it to white.
3. Add a directional source of yellow color that illuminates the scene from top downwards.
4. Set the blue color for the non-directional lighting.

```
      //--- set yellow color for the source and direct it from above downwards
      m_canvas.LightColorSet(DXColor(1.0,1.0,0.0,0.8f));
      m_canvas.LightDirectionSet(DXVector3(0.0,-1.0,0.0));
      //--- set the blue color for the ambient light
      m_canvas.AmbientColorSet(DXColor(0.0,0.0,1.0,0.4f));
      //--- create cube - pass to it the resource manager, scene parameters and coordinates of two opposite corners of the cube
      if(!m_box.Create(m_canvas.DXDispatcher(),m_canvas.InputScene(),DXVector3(-1.0,-1.0,-1.0),DXVector3(1.0,1.0,1.0)))
        {
         m_canvas.Destroy();
         return(false);
         }
      //--- set the white color for the cube
      m_box.DiffuseColorSet(DXColor(1.0,1.0,1.0,1.0));
      //--- add green glow for the cube (emission)
      m_box.EmissionColorSet(DXColor(0.0,1.0,0.0,0.2f));
```

Please note that the position of the directed light source is not set in Canvas3D, while only the direction in which light spreads out is
given. The source of the directional light is considered to be at an infinite distance and a strictly parallel light stream illuminates
the scene.

```
m_canvas.LightDirectionSet(DXVector3(0.0,-1.0,0.0));
```

Here, light spreading vector is pointed along the Y axis in the negative direction, i.e. from top downwards. Furthermore, if you set
parameters for the directed light source (LightColorSet and LightDirectionSet), you must also specify the color of the ambient light
(AmbientColorSet). By default, the color of the ambient light is set to white with maximum intensity and thus all the shadows will be
white. This means that the objects in the scene will be floodlit with white from the ambient lighting, while the directional source
light will be interrupted by the white light.

```
      //--- set yellow color for the source and direct it from above downwards
      m_canvas.LightColorSet(DXColor(1.0,1.0,0.0,0.8f));
      m_canvas.LightDirectionSet(DXVector3(0.0,-1.0,0.0));
      //--- set the blue color for the ambient light
      m_canvas.AmbientColorSet(DXColor(0.0,0.0,1.0,0.4f));  // must be specified
```

The below gif animation shows how the image changes when we add lighting. The source code of the example is available in the file _"Step6 Add_
_Light.mq5"_.

![The white cube with green emission under a yellow light source, with blue ambient light](https://c.mql5.com/2/38/box_light__1.gif)

The white cube with green emission under a yellow light source, with blue ambient
light

Try to turn off color methods in the code above to see how it works.

### Animation

Animation implies a change in scene parameters and objects over time. Any available properties can be changed depending on time or events. Set the
timer for 10 milliseconds — this event will affect the update of the scene:

```
int OnInit()
  {
...
//--- create canvas
   ExtAppWindow=new CCanvas3DWindow();
   if(!ExtAppWindow.Create(width,height))
      return(INIT_FAILED);
//--- set timer
   EventSetMillisecondTimer(10);
//---
   return(INIT_SUCCEEDED);
   }
```

Add the appropriate event handler to CCanvas3DWindow. We need to change object parameters (such as rotation, movement and zooming) and the
direction of lighting:

```
   //+------------------------------------------------------------------+
   //| Timer handler                                                    |
   //+------------------------------------------------------------------+
   void              OnTimer(void)
     {
      static ulong last_time=0;
      static float time=0;
      //--- get the current time
      ulong current_time=GetMicrosecondCount();
      //--- calculate the delta
      float deltatime=(current_time-last_time)/1000000.0f;
      if(deltatime>0.1f)
         deltatime=0.1f;
      //--- increase the elapsed time value
      time+=deltatime;
      //--- remember the time
      last_time=current_time;
      //--- calculate the cube position and the rotation matrix
      DXMatrix rotation,translation,scale;
      DXMatrixRotationYawPitchRoll(rotation,time/11.0f,time/7.0f,time/5.0f);
      DXMatrixTranslation(translation,(float)sin(time/3),0.0,0.0);
      //--- calculate the cube compression/extension along the axes
      DXMatrixScaling(scale,1.0f+0.5f*(float)sin(time/1.3f),1.0f+0.5f*(float)sin(time/1.7f),1.0f+0.5f*(float)sin(time/1.9f));
      //--- multiply the matrices to obtain the final transformation
      DXMatrix transform;
      DXMatrixMultiply(transform,scale,rotation);
      DXMatrixMultiply(transform,transform,translation);
      //--- set the transformation matrix
      m_box.TransformMatrixSet(transform);
      //--- calculate the rotation of the light source around the Z axis
      DXMatrixRotationZ(rotation,deltatime);
      DXVector3 light_direction;
      //--- get the current direction of the light source
      m_canvas.LightDirectionGet(light_direction);
      //--- calculate the new direction of the light source and set it
      DXVec3TransformCoord(light_direction,light_direction,rotation);
      m_canvas.LightDirectionSet(light_direction);
      //--- recalculate the 3D scene and draw it in the canvas
      Redraw();
      }
```

Please note that object changes are applied over initial values, as if we always deals with the initial cube state and apply all operations
related to rotation/movement/compression from scratch, which means that the current state of the cube is not saved. However, the light
source direction is changed by _deltatime_ increments from the current value.

![A rotating cube with dynamic lighting](https://c.mql5.com/2/38/box_color_flying.gif)

A rotating cube with the dynamically changing light source direction.

The result is a very complex 3D animation. The example code is available in the file _"Step7 Animation.mq5"._

### Control Camera Position Using the Mouse

Let us consider the last animation element in the 3D graphics, a reaction to user actions. Add camera management using the mouse in our
example. First, subscribe to mouse events and create the corresponding handlers:

```
int OnInit()
  {
...
//--- set the timer
   EventSetMillisecondTimer(10);
//--- enable receiving of mouse events: moving and button clicks
   ChartSetInteger(0,CHART_EVENT_MOUSE_MOVE,1);
   ChartSetInteger(0,CHART_EVENT_MOUSE_WHEEL,1)
//---
   return(INIT_SUCCEEDED);
   }
void OnDeinit(const int reason)
  {
//--- Deleting the timer
   EventKillTimer();
//--- disable the receiving of mouse events
   ChartSetInteger(0,CHART_EVENT_MOUSE_MOVE,0);
   ChartSetInteger(0,CHART_EVENT_MOUSE_WHEEL,0);
//--- delete the object
   delete ExtAppWindow;
//--- return chart to the usual display mode with price charts
   ChartSetInteger(0,CHART_SHOW,true);
   }
void OnChartEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
  {
...
//--- chart change event
   if(id==CHARTEVENT_CHART_CHANGE)
      ExtAppWindow.OnChartChange();
//--- mouse movement event
   if(id==CHARTEVENT_MOUSE_MOVE)
      ExtAppWindow.OnMouseMove((int)lparam,(int)dparam,(uint)sparam);
//--- mouse wheel scroll event
   if(id==CHARTEVENT_MOUSE_WHEEL)
      ExtAppWindow.OnMouseWheel(dparam);
```

In CCanvas3DWindow, create the mouse movement event handler. It will change camera direction angles when the mouse is moved with the left
button pressed:

```
   //+------------------------------------------------------------------+
   //| Handle mouse movements                                           |
   //+------------------------------------------------------------------+
   void              OnMouseMove(int x,int y,uint flags)
     {
      //--- left mouse button
      if((flags&1)==1)
        {
         //--- there is no information about the previous mouse position
         if(m_mouse_x!=-1)
           {
            //--- update the camera angle upon change of position
            m_camera_angles.y+=(x-m_mouse_x)/300.0f;
            m_camera_angles.x+=(y-m_mouse_y)/300.0f;
            //--- set the vertical angle in the range between (-Pi/2,Pi2)
            if(m_camera_angles.x<-DX_PI*0.49f)
               m_camera_angles.x=-DX_PI*0.49f;
            if(m_camera_angles.x>DX_PI*0.49f)
               m_camera_angles.x=DX_PI*0.49f;
            //--- update camera position
            UpdateCameraPosition();
            }
         //--- save mouse position
         m_mouse_x=x;
         m_mouse_y=y;
         }
      else
        {
         //--- reset the saved position if the left mouse button is not pressed
         m_mouse_x=-1;
         m_mouse_y=-1;
         }
      }
```

Here is the mouse wheel event handler, which changes the distance between the camera and the center of the scene:

```
   //+------------------------------------------------------------------+
   //| Handling mouse wheel events                                      |
   //+------------------------------------------------------------------+
   void              OnMouseWheel(double delta)
     {
      //--- update the distance between the camera and the center upon a mouse scroll
      m_camera_distance*=1.0-delta*0.001;
      //--- set the distance in the range between [3,50]
      if(m_camera_distance>50.0)
         m_camera_distance=50.0;
      if(m_camera_distance<3.0)
         m_camera_distance=3.0;
      //--- update camera position
      UpdateCameraPosition();
      }
```

Both handlers call the UpdateCameraPosition() method to update the camera position according to the updated parameters:

```
   //+------------------------------------------------------------------+
   //| Updates the camera position                                      |
   //+------------------------------------------------------------------+
   void              UpdateCameraPosition(void)
     {
      //--- the position of the camera taking into account the distance to the center of coordinates
      DXVector4 camera=DXVector4(0.0f,0.0f,-(float)m_camera_distance,1.0f);
      //--- camera rotation around the X axis
      DXMatrix rotation;
      DXMatrixRotationX(rotation,m_camera_angles.x);
      DXVec4Transform(camera,camera,rotation);
      //--- camera rotation around the Y axis
      DXMatrixRotationY(rotation,m_camera_angles.y);
      DXVec4Transform(camera,camera,rotation);
      //--- set camera to position
      m_canvas.ViewPositionSet(DXVector3(camera));
      }
```

The source code is available in the _"Step8 Mouse Control.mq5"_ file below.

![Control Camera Position Using the Mouse](https://c.mql5.com/2/38/box_mouse_control.gif)

Control camera position using the mouse.

### Applying Textures

A texture is a bitmap image that is applied to the surface of a polygon to represent patterns or materials. The use of textures allows
reproduce small objects on the surface, which would require too many resources if we created them using polygons. For example, this can be an
imitation of a stone, wood, soil and other materials.

CDXMesh and its child classes allow specifying texture. In the standard pixel shader this texture is used together with DiffuseColor. Remove the
object animation and apply a stone texture. It should be located in the [MQL5\\Files \\
folder](https://www.metatrader5.com/en/metaeditor/help/structure "https://www.metatrader5.com/en/metaeditor/help/structure") of the terminal working directory:

```
   virtual bool      Create(const int width,const int height)
     {
  ...
      //--- set the white color for the non-directional lighting
      m_box.DiffuseColorSet(DXColor(1.0,1.0,1.0,1.0));

      //--- add texture to draw the cube faces
      m_box.TextureSet(m_canvas.DXDispatcher(),"stone.bmp");
      //--- add the cube to the scene
      m_canvas.ObjectAdd(&m_box);
      //--- redraw the scene
      Redraw();
      //--- succeed
      return(true);
      }
```

![A cube with a stone texture](https://c.mql5.com/2/38/box_texture__1.gif)

A cube with a stone texture.

### Creating Custom Objects

All objects consist of vertices (DXVector3), which are connected into primitives
using indices. The most common primitive is a triangle. A basic 3D object is created by creating a list of vertices which contain at least
coordinates (but can also contain a lot of additional data, such as normal, color, etc.), the type of primitives into which they are
combined, and a list of vertex indices by which they will be combined into primitives.

The Standard Library has the DXVertex vertex type, which contains its coordinate, a normal for lighting calculation, texture
coordinates and color. The standard vertex shader works with this vertex type.

```
struct DXVertex
  {
   DXVector4         position;  // vertex coordinates
   DXVector4         normal;    // normal vector
   DXVector2         tcoord;    // face coordinate to apply the texture
   DXColor           vcolor;    // color
  };
```

The MQL5\\Include\\Canvas\\DXDXUtils.mqh auxiliary type contains a set of methods for generating the geometry (vertices and indices) of the
basic primitives and for loading 3D geometry from [.OBJ \\
files](https://en.wikipedia.org/wiki/Wavefront_.obj_file "https://en.wikipedia.org/wiki/Wavefront_.obj_file").

Add the creation of a sphere and a torus, apply the same stone texture:

```
   virtual bool      Create(const int width,const int height)
     {
 ...
      // --- vertices and indexes for manually created objects
      DXVertex vertices[];
      uint indices[];
      //--- prepare vertices and indices for the sphere
      if(!DXComputeSphere(0.3f,50,vertices,indices))
         return(false);
      //--- set white color for the vertices
      DXColor white=DXColor(1.0f,1.0f,1.0f,1.0f);
      for(int i=0; i<ArraySize(vertices); i++)
         vertices[i].vcolor=white;
      //--- create the sphere object
      if(!m_sphere.Create(m_canvas.DXDispatcher(),m_canvas.InputScene(),vertices,indices))
        {
         m_canvas.Destroy();
         return(false);
         }
      //--- set diffuse color for the sphere
      m_sphere.DiffuseColorSet(DXColor(0.0,1.0,0.0,1.0));
      //--- set white specular color
      m_sphere.SpecularColorSet(white);
      m_sphere.TextureSet(m_canvas.DXDispatcher(),"stone.bmp");
      //--- add the sphere to a scene
      m_canvas.ObjectAdd(&m_sphere);
      //--- prepare vertices and indices for the torus
      if(!DXComputeTorus(0.3f,0.1f,50,vertices,indices))
         return(false);
      //--- set white color for the vertices
      for(int i=0; i<ArraySize(vertices); i++)
         vertices[i].vcolor=white;
      //--- create the torus object
      if(!m_torus.Create(m_canvas.DXDispatcher(),m_canvas.InputScene(),vertices,indices))
        {
         m_canvas.Destroy();
         return(false);
         }
      //--- set diffuse color for the torus
      m_torus.DiffuseColorSet(DXColor(0.0,0.0,1.0,1.0));
      m_torus.SpecularColorSet(white);
      m_torus.TextureSet(m_canvas.DXDispatcher(),"stone.bmp");
      //--- add the torus to a scene
      m_canvas.ObjectAdd(&m_torus);
      //--- redraw the scene
      Redraw();
      //--- succeed
      return(true);
      }
```

Add animation for the new objects:

```
   void              OnTimer(void)
     {
...
      m_canvas.LightDirectionSet(light_direction);
      //--- sphere orbit
      DXMatrix translation;
      DXMatrixTranslation(translation,1.1f,0,0);
      DXMatrixRotationY(rotation,time);
      DXMatrix transform;
      DXMatrixMultiply(transform,translation,rotation);
      m_sphere.TransformMatrixSet(transform);
      //--- torus orbit with rotation around its axis
      DXMatrixRotationX(rotation,time*1.3f);
      DXMatrixTranslation(translation,-2,0,0);
      DXMatrixMultiply(transform,rotation,translation);
      DXMatrixRotationY(rotation,time/1.3f);
      DXMatrixMultiply(transform,transform,rotation);
      m_torus.TransformMatrixSet(transform);
      //--- recalculate the 3D scene and draw it in the canvas
      Redraw();
      }
```

Save changes as _Three Objects.mq5_ and run it.

![Rotating figures in the cube orbit.](https://c.mql5.com/2/38/three_objects.png)

Rotating figures in the cube orbit.

### Data-Based 3D Surface

Various graphs are usually used for creating reports and analyzing data, such as linear charts, histograms, pie diagrams, etc. MQL5 offers a
convenient [graphic library](https://www.mql5.com/en/articles/2866), which however can only build 2D charts.

The CDXSurface class allows visualizing a surface using custom data stored in a two-dimensional array. Let us view the example of the
following mathematical function

```
z=sin(2.0*pi*sqrt(x*x+y*y))
```

Create an object to draw the surface, and an array to store data:

```
   virtual bool      Create(const int width,const int height)
     {
...
      //--- prepare an array to store data
      m_data_width=m_data_height=100;
      ArrayResize(m_data,m_data_width*m_data_height);
      for(int i=0;i<m_data_width*m_data_height;i++)
         m_data[i]=0.0;
      //--- create a surface object
      if(!m_surface.Create(m_canvas.DXDispatcher(),m_canvas.InputScene(),m_data,m_data_width,m_data_height,2.0f,
                           DXVector3(-2.0,-0.5,-2.0),DXVector3(2.0,0.5,2.0),DXVector2(0.25,0.25),
                           CDXSurface::SF_TWO_SIDED|CDXSurface::SF_USE_NORMALS,CDXSurface::CS_COLD_TO_HOT))
        {
         m_canvas.Destroy();
         return(false);
         }
      //--- create texture and reflection
      m_surface.SpecularColorSet(DXColor(1.0,1.0,1.0,1.0));
      m_surface.TextureSet(m_canvas.DXDispatcher(),"checker.bmp");
      //--- add the surface to the scene
      m_canvas.ObjectAdd(&m_surface);
      //--- succeed
      return(true);
      }
```

The surface will be drawn within a box with a base of 4x4 and a height of 1. The texture dimensions are 0.25x0.25.

- SF\_TWO\_SIDED indicates that the surface will be drawn both above the surface and below it in case camera moves under the surface.
- SF\_USE\_NORMALS indicates that normal calculations will be used for calculating reflections from the surface caused by the directional light
source.
- CS\_COLD\_TO\_HOT sets the heatmap coloring of the surface from blue to red with a transition through green and yellow.


To animate the surface, add time below the sine sign and update it by timer.

```
   void              OnTimer(void)
     {
      static ulong last_time=0;
      static float time=0;
      //--- get the current time
      ulong current_time=GetMicrosecondCount();
      //--- calculate the delta
      float deltatime=(current_time-last_time)/1000000.0f;
      if(deltatime>0.1f)
         deltatime=0.1f;
      //--- increase the elapsed time value
      time+=deltatime;
      //--- remember the time
      last_time=current_time;
      //--- calculate surface values taking into account time changes
      for(int i=0; i<m_data_width; i++)
        {
         double x=2.0*i/m_data_width-1;
         int offset=m_data_height*i;
         for(int j=0; j<m_data_height; j++)
           {
            double y=2.0*j/m_data_height-1;
            m_data[offset+j]=MathSin(2.0*M_PI*sqrt(x*x+y*y)-2*time);
            }
         }
      //--- update data to draw the surface
      if(m_surface.Update(m_data,m_data_width,m_data_height,2.0f,
                          DXVector3(-2.0,-0.5,-2.0),DXVector3(2.0,0.5,2.0),DXVector2(0.25,0.25),
                          CDXSurface::SF_TWO_SIDED|CDXSurface::SF_USE_NORMALS,CDXSurface::CS_COLD_TO_HOT))
        {
         //--- recalculate the 3D scene and draw it in the canvas
         Redraw();
         }
      }
```

The source code is available in _3D Surface.mq5_, the program example is shown in the video.

YouTube

In this article, we have considered the capabilities of [DirectX functions](https://www.mql5.com/en/docs/directx)
in creating simple geometric shapes and animated 3D graphics for visual data analysis. More complex examples can be found in the MetaTrader 5
terminal installation directory: Expert Advisors "Correlation Matrix 3D" and "Math 3D Morpher", as well as the "Remnant 3D"
script.

MQL5 enables you to solve important algorithmic trading tasks without using third-party packages:

- Optimize complex trading strategies that contain many input parameters
- Obtain optimization results
- Visualize data in the most convenient three-dimensional store

Use the cutting-edge functionality to visualize stock data and to develop trading strategies in MetaTrader 5 — now with 3D graphics!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7708](https://www.mql5.com/ru/articles/7708)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7708.zip "Download all attachments in the single ZIP archive")

[Step1\_Create\_Box.mq5](https://www.mql5.com/en/articles/download/7708/step1_create_box.mq5 "Download Step1_Create_Box.mq5")(11.23 KB)

[Step2\_Rotation\_Z.mq5](https://www.mql5.com/en/articles/download/7708/step2_rotation_z.mq5 "Download Step2_Rotation_Z.mq5")(14.27 KB)

[Step3\_ViewUpDirection.mq5](https://www.mql5.com/en/articles/download/7708/step3_viewupdirection.mq5 "Download Step3_ViewUpDirection.mq5")(15.62 KB)

[Step4\_Box\_Color.mq5](https://www.mql5.com/en/articles/download/7708/step4_box_color.mq5 "Download Step4_Box_Color.mq5")(15.25 KB)

[Step5\_Box\_Translation.mq5](https://www.mql5.com/en/articles/download/7708/step5_box_translation.mq5 "Download Step5_Box_Translation.mq5")(13.62 KB)

[Step6\_Add\_Light.mq5](https://www.mql5.com/en/articles/download/7708/step6_add_light.mq5 "Download Step6_Add_Light.mq5")(14.27 KB)

[Step7\_Animation.mq5](https://www.mql5.com/en/articles/download/7708/step7_animation.mq5 "Download Step7_Animation.mq5")(18.83 KB)

[Step8\_Mouse\_Control.mq5](https://www.mql5.com/en/articles/download/7708/step8_mouse_control.mq5 "Download Step8_Mouse_Control.mq5")(24.45 KB)

[Step9\_Texture.mq5](https://www.mql5.com/en/articles/download/7708/step9_texture.mq5 "Download Step9_Texture.mq5")(23.16 KB)

[Three\_Objects.mq5](https://www.mql5.com/en/articles/download/7708/three_objects.mq5 "Download Three_Objects.mq5")(27.9 KB)

[3D\_Surface.mq5](https://www.mql5.com/en/articles/download/7708/3d_surface.mq5 "Download 3D_Surface.mq5")(24.48 KB)

[MQL5.zip](https://www.mql5.com/en/articles/download/7708/mql5.zip "Download MQL5.zip")(199.48 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/338479)**
(42)


![okwh](https://c.mql5.com/avatar/2011/9/4E7F67FD-3C19.jpg)

**[okwh](https://www.mql5.com/en/users/dxdcn)**
\|
10 May 2023 at 11:26

my display adapter is Nivada FX 1700--- an old [product](https://www.metatrader5.com/en/terminal/help/fundamental/economic_indicators_usa/usa_productivity "Productivity"). only support feture-level 10.0

So use DXcpl.exe to set Force WRAP for MT , then all run OK.

![Anatoliy Lukanin](https://c.mql5.com/avatar/2016/8/57BC3406-87A4.jpg)

**[Anatoliy Lukanin](https://www.mql5.com/en/users/luka-fx)**
\|
5 Sep 2023 at 15:30

To demonstrate the capabilities of the language, not bad.

But it is hardly useful for trading, having thought where I can use it for trading, but nothing came to my mind.

For demonstration it is better to write an Expert Advisor template with all the checks of correct opening of a pose, setting an order, modification, deletion, closing, etc., for further sending for validation, without errors.

My point is, I wrote an Expert Advisor on mt5, it trades in the terminal without errors and problems.

I sent it for validation, there are a lot of errors, for each action I wrote about 5 checks, using my own and from the

[What checks should a trading robot pass before publishing in the Market?](https://www.mql5.com/en/articles/2555)

and it was useless, I tortured myself for a month,  still a lot of errors.

I had to spit, it is not for nothing that it is hard to switch to mt5, I have no such problems with mt4.

The template will definitely be useful.

Or give me a link to a working template, I did not find a good one.

Good luck to everyone!

![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
5 Sep 2023 at 15:50

**Anatoliy Lukanin [#](https://www.mql5.com/ru/forum/337811/page4#comment_49156810):**

give me a link to a working template, I can't find a good one.

[https://www.mql5.com/ru/forum/93352/page78#comment\_48296338](https://www.mql5.com/ru/forum/93352/page78#comment_48296338)

![Anatoliy Lukanin](https://c.mql5.com/avatar/2016/8/57BC3406-87A4.jpg)

**[Anatoliy Lukanin](https://www.mql5.com/en/users/luka-fx)**
\|
5 Sep 2023 at 16:03

**fxsaber [#](https://www.mql5.com/ru/forum/337811/page4#comment_49158655):**

h [ttps://](https://www.mql5.com/ru/forum/93352/page78#comment_48296338) www.mql5.com/ru/forum/93352/page78#comment\_48296338

For the link to the thread, from the bottom of my heart!

But I'll be reading it for a week, and I don't know if I'll get it right.

I mean a link to a working template for the Expert Advisor on MT5, if there is such a template in nature.

![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
5 Sep 2023 at 16:08

**Anatoliy Lukanin [#](https://www.mql5.com/ru/forum/337811/page4#comment_49159061):**

For the link to the thread, from the bottom of my heart!

But it'll take me a week to read it, and I don't know if I'll get it right.

It's supposed to be validated.

```
#define  MT4ORDERS_AUTO_VALIDATION // Trade orders are sent only if they are successfully checked for correctness
#include <MT4Orders.mqh> // https://www.mql5.com/en/code/16006

#define  Ask SymbolInfoDouble(_Symbol, SYMBOL_ASK)

void OnTick()
{
  OrderSend(_Symbol, OP_BUY, 1, Ask, 0, 0, 0);
}
```

![Applying network functions, or MySQL without DLL: Part I - Connector](https://c.mql5.com/2/37/kisspng-computer-icons-application-programming-interface-c-database-administrator-icon-free-download.png)[Applying network functions, or MySQL without DLL: Part I - Connector](https://www.mql5.com/en/articles/7117)

MetaTrader 5 has received network functions recently. This opened up great opportunities for programmers developing products for the Market. Now they can implement things that required dynamic libraries before. In this article, we will consider them using the implementation of the MySQL as an example.

![Library for easy and quick development of MetaTrader programs (part XXXIII): Pending trading requests - closing positions under certain conditions](https://c.mql5.com/2/38/MQL5-avatar-doeasy__1.png)[Library for easy and quick development of MetaTrader programs (part XXXIII): Pending trading requests - closing positions under certain conditions](https://www.mql5.com/en/articles/7554)

We continue the development of the library functionality featuring trading using pending requests. We have already implemented sending conditional trading requests for opening positions and placing pending orders. In the current article, we will implement conditional position closure – full, partial and closing by an opposite position.

![Library for easy and quick development of MetaTrader programs (part XXXIV): Pending trading requests - removing and modifying orders and positions under certain conditions](https://c.mql5.com/2/38/MQL5-avatar-doeasy__2.png)[Library for easy and quick development of MetaTrader programs (part XXXIV): Pending trading requests - removing and modifying orders and positions under certain conditions](https://www.mql5.com/en/articles/7569)

In this article, we will complete the description of the pending request trading concept and create the functionality for removing pending orders, as well as modifying orders and positions under certain conditions. Thus, we are going to have the entire functionality enabling us to develop simple custom strategies, or rather EA behavior logic activated upon user-defined conditions.

![Applying OLAP in trading (part 3): Analyzing quotes for the development of trading strategies](https://c.mql5.com/2/38/OLAP_02.png)[Applying OLAP in trading (part 3): Analyzing quotes for the development of trading strategies](https://www.mql5.com/en/articles/7535)

In this article we will continue dealing with the OLAP technology applied to trading. We will expand the functionality presented in the first two articles. This time we will consider the operational analysis of quotes. We will put forward and test the hypotheses on trading strategies based on aggregated historical data. The article presents Expert Advisors for studying bar patterns and adaptive trading.

[![](https://www.mql5.com/ff/si/5k7a2kbftss6k97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1171%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dbest.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=nwegcasiojnqcoyrdlgofmjtfardztwf&s=d64d6f3c87f2458cba81f6d7b6694dd9e89dd354d4abc1d0584e405285806c9f&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=dlnkuxtlqgcgmjqofkjxdbgbgkunjpvy&ssn=1769192075094107674&ssn_dr=0&ssn_sr=0&fv_date=1769192075&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7708&back_ref=https%3A%2F%2Fwww.google.com%2F&title=How%20to%20create%203D%20graphics%20using%20DirectX%20in%20MetaTrader%205%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919207596518086&fz_uniq=5071685064946822283&sv=2552)

![MQL5 - Language of trade strategies built-in the MetaTrader 5 client terminal](https://c.mql5.com/i/registerlandings/logo-2.png)

You are missing trading opportunities:

- Free trading apps
- Over 8,000 signals for copying
- Economic news for exploring financial markets

RegistrationLog in

latin characters without spaces

a password will be sent to this email

An error occurred


- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup&amp;reg=1)

You agree to [website policy](https://www.mql5.com/en/about/privacy) and [terms of use](https://www.mql5.com/en/about/terms)

If you do not have an account, please [register](https://www.mql5.com/en/auth_register)

Allow the use of cookies to log in to the MQL5.com website.

Please enable the necessary setting in your browser, otherwise you will not be able to log in.

[Forgot your login/password?](https://www.mql5.com/en/auth_forgotten?return=popup)

- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)

This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).