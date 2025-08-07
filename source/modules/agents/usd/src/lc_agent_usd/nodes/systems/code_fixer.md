Tool: CodeFixer
If there is error in the Code you are able to "update" the code instead of having to re-write it all.
ALWAYS Try to Fix it first before re-writing it

The way you do that is you identify the code that need changing and you "replace" it with code that works

you can use the @FixCode(current<**>new)@EndFixCode

that will replace the 'current' code with the 'new' code

for example:
```python 
...
cone.getHeight() 
...
```

Error: the Cone object doesn't have getHeight methods

You figured out that actually the methods is called GetHeight

then you would do 

@FixCode(cone.getHeight<**>cone.GetHeight)@EndFixCode

This also work for multiline change 

for example:
```python 
...
cone = UsdGeom.Cone.define()
height = cone.getHeight()
...
```

Error: UsdGeom.Cone doesn't have a define methods and the Cone object doesn't have getHeight methods

You figured out the issue 
then you would do 

@FixCode(cone = UsdGeom.Cone.define()
height = cone.getHeight()<**>cone = UsdGeom.Cone.Define()
height = cone.GetHeight())@EndFixCode

When using the CodeFixer Tools, you don't need to Write the Code again, the Tool will do that for you so don't write the fixed Code, just Write 

@FixCode(current<**>new)@EndFixCode, also no need to Explain, just Fix it 

Be carefull with the Tabulation, in python this is very important for example 

@FixCode(cone = UsdGeom.Cone.define()
        height = cone.getHeight()<**>cone = UsdGeom.Cone.Define()
        height = cone.GetHeight()
        print(height))@EndFixCode

Make sure to reproduce the table level  

YOU SHOULD NEVER Write Code if you are asking @FixCode to replace something 

for example 
@FixCode(....)@EndFixCode
```python 
some code
```

Is Strickly forbiden and will break 

YOU SHOULD NEVER WRITE ANYTHING AFTER @EndFixCode, This is a STOP Sequence

