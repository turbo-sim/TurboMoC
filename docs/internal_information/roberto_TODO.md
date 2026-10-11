

I am preparing to do a generatization / architectural imporvement of the app.

The objective is to leave things ready for furtuer cases in which we could have

- Stator only
- Rotor only
- Stator + Rotor

In each of these 3 configurations, we could have the following options:
- Linear cascade
- Pure axial
- Radial (inflow or outflow, it does not matter as it only depends on the radii given)
- Mixed flow (we we will not create this or leave placeholder for it)

(except for the stator + rotor case, where the linar cascade should not be an option)

I think that the linear cascade is nice to keep because it lets us have the 2D stator cascade as an option from the same app

Do you think it would be simpler to always think about doing the blade-to-blade meshing in the m'-theta plane

Then, for the linear cascade case we simply specify the geometry with

- Meridionla chord (specified by the user)
- Blade height (specified by the user)
- The number of blades or radii are not needed in this linear cascade planar case

Then the coordinates are symboly taken as x=m' and y=theta and the blade is scaled to satisfy the specified chord?

Then, for the 3D case we simply extrude the blade in the spanwise direction

I am not sure if we should have the option to export the mesh in 2D and 3D, or have only 3D with the option to specify a single mesh element to make a pseudo-3d blade. Or perhaps it is better to let the user export both the 2D and 3D grids to have a simpler 2D representaiton in fluent or in other solvers that accept 2D meshes? Please give me your thoughts about this decision and tell me your suggested approach and wait for my response before implementing things in this direction.


----

Now that the app will enable stator, rotor or stator+rotor we have to think of the new design

My suggestion would be to have a selector for the design mode that we use among the 3

If we select stator only, or rotor only we could have the same design as of now

In case we select stator + rotor we will have to generalize for it. I think that we could have upper level dictionaries in the yaml for 

stator: [all content of stator nested below]
rotor: [all content of rotor nested below]

As these can have different geometries and mesh settings

Then if we agree on this point, we should reflect this hierarchy visualy on the app. I think that we could have something like the following.

- Stator
  - Geometry
    - Machine type
      - Linear cascade
      - Pure axial
      - Pure radial
  - Meshing
- Rotor
  - Geometry
  - Meshing
- Stator+Rotor
  - Stator
    - Geometry
    - Meshing
  - Rotor
    - Geometry
      - Parametrization type
        - Conventional
        - Vortex
    - Meshing


Note, that in this implementation I think it would be nice to combine the 2D and 3D meshing into the same tab/pane as the user might want to see how the 2d mesh relates to the 3D and also to be more economic in terms of spacing


A different alternative could be that we have the geometry and meshing as the only two panes and then when we are in for example the meshing pane we have

geometry
- Stator (dropdown menu)
  - All settings to define the geometry of the stator
- Rotor (dropdown menu)
  - All settings to define the geometry of the rotor

meshing
- Stator (dropdown menu)
  - All settings to define the meshing of the stator
- Rotor (dropdown menu)
  - All settings to define the mesing of the rotor


In this way it would make sense to plot both the staro and the rotor on the same figures, so we have a clear multirow view of the machine. I think that this approach could also extend in a more nicer way to multistage machines.


Please think carefully about these 2 approaches and give me your thoughts about this decision and tell me your suggested approach and wait for my response before implementing things in this direction.


------

We can leave the implementation of the rotor geometry and meshing as a placeholder, as we are not integrating with the rotor yet, but we want to leave everything ready for that integration





---


Do you advice in favor of removing all default parameters na dmechanism to handle default parameters form the source code functionality in src/
My reasoning is that these might hide unexpected behaviors. If we do not have optinal parameters we always have to specify things in the yaml file

I think it is fine to have default parameters at the app level, but in the programatic yaml file everything should be defined explicitly without hiding parameters that take defaults from the source code 

Do you agree with my interpretation or suggest to do otherwise


## GUI and UI features to keep
- We keep the configuration of the case following the yaml file on the left side of the app
- Each dictionary of the yaml file is an expandable/collapsible pane. One the relevant settings for each pane are shown in each pane, while the user has to change pane to move to see the other seetings
- Keep the workflow with different compute buttons for the different operations:
  - Compute MoC
  - Generate 2D
  - Generate 3D
- Clicking each of these buttoms should move the user to the correct pane to show the new options that the user should decide on the next step. (now the meshing 2d and meshing 3d are together, but we could still have one buttom for each operation because they take a different amount of time)
- The geometry pane should have the 3 plots in 2 expandable/scrollable
  - Blade to blade plane
    - m'-theta coordinates
    - Show the blade with shading and the flow domain
  - Meridional channel plane
    - The radius should start with radius zero to give a clear view of the blade
  - Threedimejsional blade shape
    - Slider for the number of blades shown
    - Check boxes for the surfaces shown or not
  - Blade-to-blade view on the left and the spanwise on the right (side by side)
  - 3D mesh viewer with a narriw  grid statistics plane on the right (as of now)
- When one of the compute buttons is clicked it luanches a worker and shows a loading message to the user knows that the progra is computing expensive things






- Prepare theory notes for the 2D/3D meshing
- Ask chatGPT to extend the fluent workflow from 2D to 3D
- Integrate the app with andrea. Important features to preserved from mine:




## GUI and UI features to implement
- Implement the arbitrary inlet state definition as in turbodash
- Split MoC settings into:
  - Thermodynamics
  - Nozzle design
  - Solver settings
- Design a nice logo for the app and make sure that it is the app logo (and not the plotly/dash logo) that appeas on the browser tab

## Panel design:

- MoC design
  - Thermodynamics
  - Nozzle design
  - Numerics
- Geometry
  - Meridional channel
  - Blade geometry
  - Flow domain
- Meshing
  - Blade.to-blade grid
    - numerics of the mesh and the inflation layers
  - Spanwise discretization
    - Number of layers 




App for nozzle only, app for turbomachinery only






## General app behavior

It should be able to compute:
- One stator blade
- One stator + One rotor blade


What steps would be needed

- Meanline calculation (simple)
- Design of the stator
  - Conventional parametrization
  - Method of characteristics
- Design of the rotor
  - Coventional parametrization
  - Vortex method


Should the parametrization of the stator always be MoC? or let the user choose the conventional one?

The user should be able to select the parametrization type for the rotor


Geometry definition should be decoupled from mehsing. Any geometry parametrization should be accepted by the mesher



First choice
- Nozzle (we do not implement today)
- Stator
  - Geometry
    - Machine type
      - Linear cascade
      - Pure axial
      - Pure radial
      - Mixed flow (not today)
  - Meshing
- Rotor
  - Geometry
  - Meshing
- Stator+Rotor
  - Stator
    - Geometry
    - Meshing
  - Rotor
    - Geometry
      - Parametrization type
        - Conventional
        - Vortex
    - Meshing



Rober


Inlet thermodynamikc pair

Add one option to have mach number or backpressure

Solver and sonic line options can be dropdown menues



Simplify the yt and rho_d definition so we provide only the ratio of thse, and have an intuitive explanation of what this ratio means


Pin as reference feature. We can keep in the geometrical views to bcompare with the previous design


Keep the option to symmetric and assumetric blading

Andrea has the option to pin a geometric design

Andrea has the option to change the control point positions

Andrea is scaling thing sup using the mass flow rate


Andrea implemented a pitch-preserving parametrization
Roberto does not have a pitch preserving parametrization



We should use andreas UX that when a variable is forzen by another, the input box gets shaded and shows the computed value and does not let the user change it

If a download button is not ready because the file are not generated it should shade the clicking box.







----

I have made a merging of my repository with a new app structure. I now have two apps. The legacy version and the new version. you can find them here:

'c:/Users/roagr/OneDrive - Danmarks Tekniske Universitet/Roberto/git_projects/method_of_characteristics/examples/apps'

Now, the new app is very nice and sleek but it has some deficienties. The biggest defifciency at this point is that it is missing the functionality for the rotor blade design. Could you please implement the functionality to design the rotor that was available in the lecacy version of the app, but adhering to the new version of the app? There should be a nice implementation for the interspace gap between stator and rotor


----

What is the purpose of the "export" mesh buttoms in the app?

When generating meshes in 2D or 3D they should not be exported in .msh format, only in CGNS format already adapted to work well with fluent as this is the intended workflow.



---

I have to consolidate the exporting so it exports a zip file with all files or some instructions because now I have to do click on a long list of files which is cumbersome

