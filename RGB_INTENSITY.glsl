#version 150
// Copyright 2015, Christopher J. Foster and the other displaz contributors.
// Use of this code is governed by the BSD-style license found in LICENSE.txt

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 modelViewProjectionMatrix;

//------------------------------------------------------------------------------
#if defined(VERTEX_SHADER)

uniform float pointRadius = 0.1;   //# uiname=Point Radius; min=0.001; max=200
uniform float trimRadius = 1000000;//# uiname=Trim Radius; min=1; max=1000000
uniform float exposure = 1.0;      //# uiname=Exposure; min=0.01; max=10000
uniform float contrast = 1.0;      //# uiname=Contrast; min=0.01; max=10000
uniform int colorMode = 0;         //# uiname=Colour Mode; enum=Intensity|Colour|Return Index|Point Source|Las Classification|File Number|Entropy
uniform int selectionMode = 0;     //# uiname=Selection; enum=All|Classified|First Return|Last Return|First Of Several|Class0|Class1|Class2|Class3|Class4|Class5|Class6|Class7|Class8|Class9|Class10|Class11|Class12|Intensity[000,100)|Intensity[100,200)|Intensity[200,300)|Intensity[300,400)|Intensity[400,500)|Intensity[500,600)|Intensity[600,700)|Intensity[700,800)|Intensity[800,900)|Intensity[900,1000]
uniform float minPointSize = 0;
uniform float maxPointSize = 400.0;
// Point size multiplier to get from a width in projected coordinates to the
// number of pixels across as required for gl_PointSize
uniform float pointPixelScale = 0;
uniform vec3 cursorPos = vec3(0);
uniform int fileNumber = 0;
in float intensity;
in vec3 position;
in vec3 color;
in int returnNumber;
in int numberOfReturns;
in int pointSourceId;
in int classification;
in float ent;
//in float heightAboveGround;

flat out float modifiedPointRadius;
flat out float pointScreenSize;
flat out vec3 pointColor;
flat out int markerShape;

float tonemap(float x, float exposure, float contrast)
{
    float Y = pow(exposure*x, contrast);
    Y = Y / (1.0 + Y);
    return Y;
}

vec3 jet_colormap(float x)
{
    if (x < 0.125)
        return vec3(0, 0, 0.5 + 4*x);
    if (x < 0.375)
        return vec3(0, 4*(x-0.125), 1);
    if (x < 0.625)
        return vec3(4*(x-0.375), 1, 1 - 4*(x-0.375));
    if (x < 0.875)
        return vec3(1, 1 - 4*(x-0.625), 0);
    return vec3(1 - 4*(x-0.875), 0, 0);
}

void main()
{
    vec4 p = modelViewProjectionMatrix * vec4(position,1.0);
    float r = length(position.xy - cursorPos.xy);
    float trimFalloffLen = min(5, trimRadius/2);
    float trimScale = min(1, (trimRadius - r)/trimFalloffLen);
    modifiedPointRadius = pointRadius * trimScale;
    pointScreenSize = clamp(2*pointPixelScale*modifiedPointRadius / p.w, minPointSize, maxPointSize);
    markerShape = 1;
    // Compute vertex color
    if (colorMode == 0)
        pointColor = vec3((intensity/1000)*(2*(intensity/1000)-1),4*(intensity/1000)*(1-(intensity/1000)),(1-(intensity/1000))*(1-2*(intensity/1000)));
    else if (colorMode == 1)
        pointColor = contrast*(exposure*color - vec3(0.5)) + vec3(0.5);
    else if (colorMode == 2)
        pointColor = vec3(0.2*returnNumber*exposure, 0.2*numberOfReturns*exposure, 0);
    else if (colorMode == 3)
    {
        markerShape = (pointSourceId+1) % 5;
        vec3 cols[] = vec3[](vec3(1,1,1), vec3(1,0,0), vec3(0,1,0), vec3(0,0,1),
                             vec3(1,1,0), vec3(1,0,1), vec3(0,1,1));
        pointColor = cols[(pointSourceId+3) % 7];
    }
    else if (colorMode == 4)
    {
        // Colour according to some common classifications defined in the LAS spec
        if (classification == 0)      pointColor = vec3(0.0, 0.0, 0.0); // ground
        else if (classification == 1) pointColor = vec3(0.0, 0.0, 1.0); // low vegetation
        else if (classification == 2) pointColor = vec3(0.0, 1.0, 0.0); // medium vegetation
        else if (classification == 3) pointColor = vec3(1.0, 0.0,  0.0); // high vegetation
        else if (classification == 4) pointColor = vec3(0.0,  0.5,  0.5); // building
        else if (classification == 5) pointColor = vec3(1.0,  0.0,  1.0); // water
        else if (classification == 10) pointColor = vec3(1.0,  0.2,  0.1); // water
    }
    else if (colorMode == 5)
    {
        // Set point colour and marker shape cyclically based on file number
        markerShape = fileNumber % 5;
        pointColor = vec3((1.0/2.0) * (0.5 + (fileNumber % 2)),
                          (1.0/3.0) * (0.5 + (fileNumber % 3)),
                          (1.0/5.0) * (0.5 + (fileNumber % 5)));
    }
    else if (colorMode == 6)
    {
        // Set point colour and marker shape cyclically based on file number
        markerShape = fileNumber % 5;
        pointColor = vec3(ent*(2*ent-1),4*ent*(1-ent),(1-ent)*(1-2*ent));
    }
    /*
    else if (colorMode == 8)
    {
        // Color based on height above ground
        pointColor = 0.8*jet_colormap(tonemap(0.16*heightAboveGround, exposure, 3.8*contrast));
    }
    */
    if (selectionMode != 0)
    {
        if (selectionMode == 1)
        {
            if (classification == 0)
                markerShape = -1;
        }
        else if (selectionMode == 2)
        {
            if (returnNumber != 1)
                markerShape = -1;
        }
        else if (selectionMode == 3)
        {
            if (returnNumber != numberOfReturns)
                markerShape = -1;
        }
        else if (selectionMode == 4)
        {
            if (returnNumber != 1 || numberOfReturns < 2)
                markerShape = -1;
        }
        else if (selectionMode == 5)
        // Intensity USER Edit Keep
        {
            if (classification != 6)
                markerShape = -1;
        }
        else if (selectionMode == 6)
        // Intensity USER Edit Keep
        {
            if (classification != 1)
                markerShape = -1;
        }
        else if (selectionMode == 7)
        // Intensity USER Edit Keep
        {
            if (classification != 2)
                markerShape = -1;
        }
        else if (selectionMode == 8)
        // Intensity USER Edit Keep
        {
            if (classification != 3)
                markerShape = -1;
        }
        else if (selectionMode == 9)
        // Intensity USER Edit Keep
        {
            if (classification != 4)
                markerShape = -1;
        }
        else if (selectionMode == 10)
        // Intensity USER Edit Keep
        {
            if (classification != 5)
                markerShape = -1;
        }
        else if (selectionMode == 11)
        // Intensity USER Edit Keep
        {
            if (classification != 6)
                markerShape = -1;
        }
        else if (selectionMode == 12)
        // Intensity USER Edit Keep
        {
            if (classification != 7)
                markerShape = -1;
        }
        else if (selectionMode == 13)
        // Intensity USER Edit Keep
        {
            if (classification != 8)
                markerShape = -1;
        }
        else if (selectionMode == 14)
        // Intensity USER Edit Keep
        {
            if (classification != 9)
                markerShape = -1;
        }
        else if (selectionMode == 15)
        // Intensity USER Edit Keep
        {
            if (classification != 10)
                markerShape = -1;
        }
        else if (selectionMode == 16)
        // Intensity USER Edit Keep
        {
            if (classification != 11)
                markerShape = -1;
        }
        else if (selectionMode == 17)
        // Intensity USER Edit Keep
        {
            if (classification != 12)
                markerShape = -1;
        }
        else if (selectionMode == 18)
        // Intensity USER Edit Keep
        {
            if (intensity < 0||intensity >= 100)
                markerShape = -1;
        }
        else if (selectionMode == 19)
        // Intensity USER Edit Keep
        {
            if (intensity < 100||intensity >= 200)
                markerShape = -1;
        }
        else if (selectionMode == 20)
        // Intensity USER Edit Keep
        {
            if (intensity < 200||intensity >= 300)
                markerShape = -1;
        }
        else if (selectionMode == 21)
        // Intensity USER Edit Keep
        {
            if (intensity < 300||intensity >= 400)
                markerShape = -1;
        }
        else if (selectionMode == 22)
        // Intensity USER Edit Keep
        {
            if (intensity < 400||intensity >= 500)
                markerShape = -1;
        }
        else if (selectionMode == 23)
        // Intensity USER Edit Keep
        {
            if (intensity < 500||intensity >= 600)
                markerShape = -1;
        }
        else if (selectionMode == 24)
        // Intensity USER Edit Keep
        {
            if (intensity < 600||intensity >= 700)
                markerShape = -1;
        }
        else if (selectionMode == 25)
        // Intensity USER Edit Keep
        {
            if (intensity < 700||intensity >= 700)
                markerShape = -1;
        }
        else if (selectionMode == 26)
        // Intensity USER Edit Keep
        {
            if (intensity < 800||intensity >= 900)
                markerShape = -1;
        }
        else if (selectionMode == 27)
        // Intensity USER Edit Keep
        {
            if (intensity < 900||intensity > 1000)
                markerShape = -1;
        }
    }
    // Ensure zero size points are discarded.  The actual minimum point size is
    // hardware and driver dependent, so set the markerShape to discarded for
    // good measure.
    if (pointScreenSize <= 0)
    {
        pointScreenSize = 0;
        markerShape = -1;
    }
    else if (pointScreenSize < 1)
    {
        // Clamp to minimum size of 1 to avoid aliasing with some drivers
        pointScreenSize = 1;
    }
    gl_PointSize = pointScreenSize;
    gl_Position = p;
}


//------------------------------------------------------------------------------
#elif defined(FRAGMENT_SHADER)

uniform float markerWidth = 0.3;

flat in float modifiedPointRadius;
flat in float pointScreenSize;
flat in vec3 pointColor;
flat in int markerShape;

out vec4 fragColor;

// Limit at which the point is rendered as a small square for antialiasing
// rather than using a specific marker shape
const float pointScreenSizeLimit = 2;
const float sqrt2 = 1.414213562;

void main()
{
    if (markerShape < 0) // markerShape == -1: discarded.
        discard;
    // (markerShape == 0: Square shape)
#   ifndef BROKEN_GL_FRAG_COORD
    gl_FragDepth = gl_FragCoord.z;
#   endif
    if (markerShape > 0 && pointScreenSize > pointScreenSizeLimit)
    {
        float w = markerWidth;
        if (pointScreenSize < 2*pointScreenSizeLimit)
        {
            // smoothly turn on the markers as we get close enough to see them
            w = mix(1, w, pointScreenSize/pointScreenSizeLimit - 1);
        }
        vec2 p = 2*(gl_PointCoord - 0.5);
        if (markerShape == 1) // shape: .
        {
            float r = length(p);
            if (r > 1)
                discard;
#           ifndef BROKEN_GL_FRAG_COORD
            gl_FragDepth += projectionMatrix[3][2] * gl_FragCoord.w*gl_FragCoord.w
                            // TODO: Why is the factor of 0.5 required here?
                            * 0.5*modifiedPointRadius*sqrt(1-r*r);
#           endif
        }
        else if (markerShape == 2) // shape: o
        {
            float r = length(p);
            if (r > 1 || r < 1 - w)
                discard;
        }
        else if (markerShape == 3) // shape: x
        {
            w *= 0.5*sqrt2;
            if (abs(p.x + p.y) > w && abs(p.x - p.y) > w)
                discard;
        }
        else if (markerShape == 4) // shape: +
        {
            w *= 0.5;
            if (abs(p.x) > w && abs(p.y) > w)
                discard;
        }
    }
    fragColor = vec4(pointColor, 1);
}

#endif


