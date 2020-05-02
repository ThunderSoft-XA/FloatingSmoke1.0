#include <math.h>
#include <iostream>

#include <stdlib.h>
#include <sstream> 
#include <fstream>
#ifdef _OGLES3
#include "OpenGLES/FrmGLES3.h"
//#include <GLES3/gl3.h>
//#include <GLES2/gl2ext.h>
#else
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <GLES3/gl3.h>
#endif

#include <glm/glm.hpp>
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtc/matrix_transform.hpp"
//#include "glm/detail/func_geometric.hpp"
#include "glm/geometric.hpp"

using namespace std;

extern ofstream g_logfile;
#include "fluid.h"





#ifndef GL_ES
#define GL_ES
#endif

//#define GRADIENT_COLOR

#ifdef GL_ES
#define STRINGIZE(shader)	"#version 300 es\n" "#pragma debug(on)\n" "precision mediump float;\n" #shader
//#define STRINGIZE(shader)	"#version 300 es\n" #shader
#else
#define STRINGIZE(shader)	"#version 330 core\n" #shader
#endif

//{{ Vertex Shaders
const GLchar *vs_fluid = STRINGIZE(
	in vec4 Position;

	void main()
	{
		gl_Position = Position;
	}
);
//}} Vertex Shaders


//{{ Fragment Shaders
const GLchar * fs_fill = STRINGIZE(
	out vec3 FragColor;

	void main()
	{
		FragColor = vec3(1.0f, 0.0f, 0.0f);
	}
);


const GLchar * fs_advect = STRINGIZE(
	out vec4 FragColor;

	uniform sampler2D VelocityTexture;
	uniform sampler2D SourceTexture;
	uniform sampler2D Obstacles;

	uniform vec2 InverseSize;
	uniform float TimeStep;
	uniform float Dissipation;

	void main()
	{
		vec2 fragCoord = gl_FragCoord.xy;
		float solid = texture(Obstacles, InverseSize * fragCoord).x;

		if (solid > 0.0f)
		{
			FragColor = vec4(0.0f);
			return;
		}

		vec2 u = texture(VelocityTexture, InverseSize * fragCoord).xy;
		vec2 coord = InverseSize * (fragCoord - TimeStep * u);
		FragColor = Dissipation * texture(SourceTexture, coord);
	}
);


const GLchar * fs_jacobi = STRINGIZE(
	out vec4 FragColor;

	uniform sampler2D Pressure;
	uniform sampler2D Divergence;
	uniform sampler2D Obstacles;

	uniform float Alpha;
	uniform float InverseBeta;

	void main()
	{
		ivec2 T = ivec2(gl_FragCoord.xy);

		// Find neighboring pressure:
		vec4 pN = texelFetchOffset(Pressure, T, 0, ivec2(0, 1));
		vec4 pS = texelFetchOffset(Pressure, T, 0, ivec2(0, -1));
		vec4 pE = texelFetchOffset(Pressure, T, 0, ivec2(1, 0));
		vec4 pW = texelFetchOffset(Pressure, T, 0, ivec2(-1, 0));
		vec4 pC = texelFetch(Pressure, T, 0);

		// Find neighboring obstacles:
		vec3 oN = texelFetchOffset(Obstacles, T, 0, ivec2(0, 1)).xyz;
		vec3 oS = texelFetchOffset(Obstacles, T, 0, ivec2(0, -1)).xyz;
		vec3 oE = texelFetchOffset(Obstacles, T, 0, ivec2(1, 0)).xyz;
		vec3 oW = texelFetchOffset(Obstacles, T, 0, ivec2(-1, 0)).xyz;

		// Use center pressure for solid cells:
		if (oN.x > 0.0f) pN = pC;
		if (oS.x > 0.0f) pS = pC;
		if (oE.x > 0.0f) pE = pC;
		if (oW.x > 0.0f) pW = pC;

		vec4 bC = texelFetch(Divergence, T, 0);
		FragColor = (pW + pE + pS + pN + Alpha * bC) * InverseBeta;
	}
);


const GLchar * fs_subtract_gradient = STRINGIZE(
	out vec2 FragColor;

	uniform sampler2D Velocity;
	uniform sampler2D Pressure;
	uniform sampler2D Obstacles;
	uniform float GradientScale;

	void main()
	{
		ivec2 T = ivec2(gl_FragCoord.xy);

		vec3 oC = texelFetch(Obstacles, T, 0).xyz;
		if (oC.x > 0.0f)
		{
			FragColor = oC.yz;
			return;
		}

		// Find neighboring pressure:
		float pN = texelFetchOffset(Pressure, T, 0, ivec2(0, 1)).r;
		float pS = texelFetchOffset(Pressure, T, 0, ivec2(0, -1)).r;
		float pE = texelFetchOffset(Pressure, T, 0, ivec2(1, 0)).r;
		float pW = texelFetchOffset(Pressure, T, 0, ivec2(-1, 0)).r;
		float pC = texelFetch(Pressure, T, 0).r;

		// Find neighboring obstacles:
		vec3 oN = texelFetchOffset(Obstacles, T, 0, ivec2(0, 1)).xyz;
		vec3 oS = texelFetchOffset(Obstacles, T, 0, ivec2(0, -1)).xyz;
		vec3 oE = texelFetchOffset(Obstacles, T, 0, ivec2(1, 0)).xyz;
		vec3 oW = texelFetchOffset(Obstacles, T, 0, ivec2(-1, 0)).xyz;

		// Use center pressure for solid cells:
		vec2 obstV = vec2(0);
		vec2 vMask = vec2(1);

		if (oN.x > 0.0f) { pN = pC; obstV.y = oN.z; vMask.y = 0.0f; }
		if (oS.x > 0.0f) { pS = pC; obstV.y = oS.z; vMask.y = 0.0f; }
		if (oE.x > 0.0f) { pE = pC; obstV.x = oE.y; vMask.x = 0.0f; }
		if (oW.x > 0.0f) { pW = pC; obstV.x = oW.y; vMask.x = 0.0f; }

		// Enforce the free-slip boundary condition:
		vec2 oldV = texelFetch(Velocity, T, 0).xy;
		vec2 grad = vec2(pE - pW, pN - pS) * GradientScale;
		vec2 newV = oldV - grad;
		FragColor = (vMask * newV) + obstV;
	}
);


const GLchar * fs_compute_divergence = STRINGIZE(
	out float FragColor;

	uniform sampler2D Velocity;
	uniform sampler2D Obstacles;
	uniform float HalfInverseCellSize;

	void main()
	{
		ivec2 T = ivec2(gl_FragCoord.xy);

		// Find neighboring velocities:
		vec2 vN = texelFetchOffset(Velocity, T, 0, ivec2(0, 1)).xy;
		vec2 vS = texelFetchOffset(Velocity, T, 0, ivec2(0, -1)).xy;
		vec2 vE = texelFetchOffset(Velocity, T, 0, ivec2(1, 0)).xy;
		vec2 vW = texelFetchOffset(Velocity, T, 0, ivec2(-1, 0)).xy;

		// Find neighboring obstacles:
		vec3 oN = texelFetchOffset(Obstacles, T, 0, ivec2(0, 1)).xyz;
		vec3 oS = texelFetchOffset(Obstacles, T, 0, ivec2(0, -1)).xyz;
		vec3 oE = texelFetchOffset(Obstacles, T, 0, ivec2(1, 0)).xyz;
		vec3 oW = texelFetchOffset(Obstacles, T, 0, ivec2(-1, 0)).xyz;

		// Use obstacle velocities for solid cells:
		if (oN.x > 0.0f) vN = oN.yz;
		if (oS.x > 0.0f) vS = oS.yz;
		if (oE.x > 0.0f) vE = oE.yz;
		if (oW.x > 0.0f) vW = oW.yz;

		FragColor = HalfInverseCellSize * (vE.x - vW.x + vN.y - vS.y);
	}
);


const GLchar * fs_splat = STRINGIZE(
	out vec4 FragColor;

	uniform vec2 Point;
	uniform vec3 FillColor;
	uniform float Radius;

	void main()
	{
		float d = distance(Point, gl_FragCoord.xy);
		if (d < Radius)
		{
			float a = (Radius - d) * 0.5f;

			a = min(a, 1.0f);
			FragColor = vec4(FillColor, a);
		}
		else
		{
			FragColor = vec4(0.0f);
		}
	}
);


const GLchar * fs_buoyancy = STRINGIZE(
	out vec2 FragColor;

	uniform sampler2D Velocity;
	uniform sampler2D Temperature;
	uniform sampler2D Density;
	uniform float AmbientTemperature;
	uniform float TimeStep;
	uniform float Sigma;
	uniform float Kappa;

	void main()
	{
		ivec2 TC = ivec2(gl_FragCoord.xy);
		float T = texelFetch(Temperature, TC, 0).r;
		vec2 V = texelFetch(Velocity, TC, 0).xy;

		FragColor = V;

		if (T > AmbientTemperature)
		{
			float D = texelFetch(Density, TC, 0).x;
			FragColor += (TimeStep * (T - AmbientTemperature) * Sigma - D * Kappa ) * vec2(0.0f, 1.0f);
		}
	}
);


const GLchar * fs_visualize = STRINGIZE(
	out vec4 FragColor;

	uniform sampler2D Sampler;
	uniform vec3 FillColor;
	uniform vec2 Scale;

	uniform int		EnableOffset;
	uniform vec2	Offset;
	uniform vec2	Viewport;

	void main()
	{
		vec2 coord = gl_FragCoord.xy;

		if (EnableOffset == 1)
		{
			coord -= Offset;
		}

		float L = texture(Sampler, coord * Scale).r;

		FragColor = vec4(FillColor, L);
	}
);
//}} Fragment Shaders





static int ViewportWidth	= 960;
static int ViewportHeight	= 960;

static int ScreenWidth	= 600;
static int ScreenHeight	= 900;

static int TextureWidth()		{ return (ViewportWidth / 2); }
static int TextureHeight()		{ return (ViewportHeight / 2); }

static int SplatRadius()	{ return ((float) TextureWidth() / 8.0f); }

#define CellSize (1.25f)

static const float	AmbientTemperature = 0.0f;
static const float	ImpulseTemperature = 10.0f;
static const float	ImpulseDensity = 1.0f;
static const int	NumJacobiIterations = 40;
static const float	TimeStep = 0.15f;
static const float	SmokeBuoyancy = 2.0f;
static const float	SmokeWeight = 0.3f;
static const float	GradientScale = 1.125f / CellSize;
static const float	TemperatureDissipation = 0.99f;
static const float	VelocityDissipation = 0.99f;
static const float	DensityDissipation = 0.9999f;
static const int	PositionSlot = 0;



struct ProgramsRec {
    GLuint Advect;
    GLuint Jacobi;
    GLuint SubtractGradient;
    GLuint ComputeDivergence;
    GLuint ApplyImpulse;
    GLuint ApplyBuoyancy;
} Programs;



typedef struct Surface_ {
    GLuint FboHandle;
    GLuint TextureHandle;
    int NumComponents;
} Surface;

typedef struct Slab_ {
    Surface Ping;
    Surface Pong;
} Slab;


typedef struct Vector2_ {
    int X;
    int Y;
} Vector2;



static GLuint QuadVao;
static GLuint VisualizeProgram;
static Slab Velocity, Density, Pressure, Temperature;
static Surface Divergence, Obstacles, HiresObstacles;



auto GetAnchorPoint = [](int background_width, int backgroud_height, int viewport_width, int viewport_height)
		{
			Vector2 point = { (background_width - viewport_width)/2, (backgroud_height - viewport_height)/2 };
			return point;
		};



GLuint CreateProgram(const GLchar * vs_src, const GLchar * gs_src, const GLchar * fs_src)
{
	GL_CHECK_CONDITION(vs_src != 0, "Vertex shader is missing!");

    GLint	compiled;
    GLchar	compilerSpew[256];
    GLuint	programHandle = glCreateProgram();
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    GLuint vsHandle = glCreateShader(GL_VERTEX_SHADER);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    glShaderSource(vsHandle, 1, &vs_src, 0);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glCompileShader(vsHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glGetShaderiv(vsHandle, GL_COMPILE_STATUS, &compiled);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glGetShaderInfoLog(vsHandle, sizeof(compilerSpew), 0, compilerSpew);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    GL_CHECK_CONDITION(compiled, compilerSpew);
    glAttachShader(programHandle, vsHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    GLuint gsHandle;
    if (gs_src)
    {/*
        gsHandle = glCreateShader(GL_GEOMETRY_SHADER);

        glShaderSource(gsHandle, 1, &gs_src, 0);
        glCompileShader(gsHandle);
        glGetShaderiv(gsHandle, GL_COMPILE_STATUS, &compiled);
        glGetShaderInfoLog(gsHandle, sizeof(compilerSpew), 0, compilerSpew);
        GL_CHECK_CONDITION(compiled, compilerSpew);
        glAttachShader(programHandle, gsHandle);*/
    }

    GLuint fsHandle;
    if (fs_src)
    {
        fsHandle = glCreateShader(GL_FRAGMENT_SHADER);
        GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
        glShaderSource(fsHandle, 1, &fs_src, 0);
        GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
        glCompileShader(fsHandle);
        GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
        glGetShaderiv(fsHandle, GL_COMPILE_STATUS, &compiled);
        GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
        glGetShaderInfoLog(fsHandle, sizeof(compilerSpew), 0, compilerSpew);
        GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
        GL_CHECK_CONDITION(compiled, compilerSpew);
        glAttachShader(programHandle, fsHandle);
        GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    }

    glLinkProgram(programHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    GLint linkSuccess;
    glGetProgramiv(programHandle, GL_LINK_STATUS, &linkSuccess);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glGetProgramInfoLog(programHandle, sizeof(compilerSpew), 0, compilerSpew);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    if (!linkSuccess)
    {
        LOG_ERROR("GL link error", compilerSpew);
        if (vs_src) LOG_INFO("Vertex Shader:", vs_src);
        if (gs_src) LOG_INFO("Geometry Shader:", gs_src);
        if (fs_src) LOG_INFO("Fragment Shader:", fs_src);
    }

    return programHandle;
}





GLuint CreateQuad()
{
    short positions[] = {
        -1, -1,
         1, -1,
        -1,  1,
         1,  1,
    };

    // Create the VAO:
    GLuint vao;
    glGenVertexArrays(1, &vao);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindVertexArray(vao);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    // Create the VBO:
    GLuint vbo;
    GLsizeiptr size = sizeof(positions);
    glGenBuffers(1, &vbo);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBufferData(GL_ARRAY_BUFFER, size, positions, GL_STATIC_DRAW);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    // Set up the vertex layout:
    GLsizeiptr stride = 2 * sizeof(positions[0]);
    glEnableVertexAttribArray(PositionSlot);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glVertexAttribPointer(PositionSlot, 2, GL_SHORT, GL_FALSE, stride, 0);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    return vao;
}






void CreateObstacles(Surface dest, int width, int height)
{
	PRINT_SEPARATOR();
    LOG_INFO("Width:", width);
    LOG_INFO("Height:", height);
    PRINT_SEPARATOR();

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
	glViewport(0, 0, width, height);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glClearColor(0, 0, 0, 0);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glClear(GL_COLOR_BUFFER_BIT);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    GLuint vao;
    glGenVertexArrays(1, &vao);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindVertexArray(vao);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    GLuint program = CreateProgram(vs_fluid, 0, fs_fill);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUseProgram(program);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    const int DrawBorder = 1;
    if (DrawBorder)
    {
        //#define T 0.9999f
		#define W 0.9999f
		#define H 0.9999f
		#define C 1.0f
        float positions[] = { -W, -H, W/C, -H, W/C,  H, -W,  H, -W, -H };
        #undef W
		#undef H

        GLuint vbo;
        GLsizeiptr size = sizeof(positions);

        glGenBuffers(1, &vbo);
        GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
        glBufferData(GL_ARRAY_BUFFER, size, positions, GL_STATIC_DRAW);
        GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

        GLsizeiptr stride = 2 * sizeof(positions[0]);

        glEnableVertexAttribArray(PositionSlot);
        GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
        glVertexAttribPointer(PositionSlot, 2, GL_FLOAT, GL_FALSE, stride, 0);
        GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
        glDrawArrays(GL_LINE_STRIP, 0, 5);
        GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
        glDeleteBuffers(1, &vbo);
        GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    }

    const int DrawCircle = 1;
    if (DrawCircle)
    {
        const int slices = 64;//1;//
        float positions[slices*2*3];
        float twopi = 8*atan(1.0f);
        float theta = 0;
        float dtheta = twopi / (float) (slices - 1);

        float coef = 0.16f;
        float* pPositions = &positions[0];
        float W = width;//ScreenWidth;
        float H = height;//ScreenHeight;
        for (int i = 0; i < slices; i++)
        {
            *pPositions++ = 0.0f;//0;
            *pPositions++ = 0.0f;//0;

            *pPositions++ = coef * cos(theta) * H / W;
            *pPositions++ = coef * sin(theta);
            theta += dtheta;

            *pPositions++ = coef * cos(theta) * H / W;
            *pPositions++ = coef * sin(theta);
        }

        GLuint vbo;
        GLsizeiptr size = sizeof(positions);

        glGenBuffers(1, &vbo);
        GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
        glBufferData(GL_ARRAY_BUFFER, size, positions, GL_STATIC_DRAW);
        GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

        GLsizeiptr stride = 2 * sizeof(positions[0]);

        //std::cout << "Size: " << size << ", ElementNumber: " << size/sizeof(positions[0]) << ", Stride: " << stride << endl;

        glEnableVertexAttribArray(PositionSlot);
        GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
        glVertexAttribPointer(PositionSlot, 2, GL_FLOAT, GL_FALSE, stride, 0);
        GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
        glDrawArrays(GL_TRIANGLES, 0, slices * 3);
        GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
        glDeleteBuffers(1, &vbo);
        GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    }

    // Cleanup
    glDeleteProgram(program);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glDeleteVertexArrays(1, &vao);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
}





Surface CreateSurface(GLsizei width, GLsizei height, int numComponents)
{
    GLuint fboHandle;

    glGenFramebuffers(1, &fboHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindFramebuffer(GL_FRAMEBUFFER, fboHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    GLuint textureHandle;

    glGenTextures(1, &textureHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindTexture(GL_TEXTURE_2D, textureHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    const int UseHalfFloats = 1;
    if (UseHalfFloats)
    {
        switch (numComponents)
        {
            case 1: glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, width, height, 0, GL_RED, GL_HALF_FLOAT, 0); 
            		GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");break;
            case 2: glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, width, height, 0, GL_RG, GL_HALF_FLOAT, 0);
            		GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR"); break;
            case 3: glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_HALF_FLOAT, 0);
            		GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR"); break;
            case 4: glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_HALF_FLOAT, 0);
            		GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR"); break;
            default: LOG_ERROR("Slab", "Illegal slab format.");
        }
    }
    else
    {
        switch (numComponents)
        {
            case 1: glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, 0);
            		GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR"); break;
            case 2: glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width, height, 0, GL_RG, GL_FLOAT, 0);
            		GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR"); break;
            case 3: glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, 0);
            		GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR"); break;
            case 4: glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, 0);
            		GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR"); break;
            default: LOG_ERROR("Slab", "Illegal slab format.");
        }
    }

    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "Unable to create normals texture");

    GLuint colorbuffer;

    glGenRenderbuffers(1, &colorbuffer);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindRenderbuffer(GL_RENDERBUFFER, colorbuffer);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureHandle, 0);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "Unable to attach color buffer");
    GL_CHECK_CONDITION(GL_FRAMEBUFFER_COMPLETE == glCheckFramebufferStatus(GL_FRAMEBUFFER), "Unable to create FBO.");

    Surface surface = { fboHandle, textureHandle, numComponents };

    glClearColor(0, 0, 0, 0);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glClear(GL_COLOR_BUFFER_BIT);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    return surface;
}


Slab CreateSlab(GLsizei width, GLsizei height, int numComponents)
{
    Slab slab;

    slab.Ping = CreateSurface(width, height, numComponents);
    slab.Pong = CreateSurface(width, height, numComponents);

    return slab;
}


static void ResetState()
{
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, 0);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, 0);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glDisable(GL_BLEND);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
}



void InitSlabOps()
{
    Programs.Advect = CreateProgram(vs_fluid, 0, fs_advect);
    Programs.Jacobi = CreateProgram(vs_fluid, 0, fs_jacobi);
    Programs.SubtractGradient = CreateProgram(vs_fluid, 0, fs_subtract_gradient);
    Programs.ComputeDivergence = CreateProgram(vs_fluid, 0, fs_compute_divergence);
    Programs.ApplyImpulse = CreateProgram(vs_fluid, 0, fs_splat);
    Programs.ApplyBuoyancy = CreateProgram(vs_fluid, 0, fs_buoyancy);/**/
}

void SwapSurfaces(Slab* slab)
{
    Surface temp = slab->Ping;
    slab->Ping = slab->Pong;
    slab->Pong = temp;
}

void ClearSurface(Surface s, float v)
{
    glBindFramebuffer(GL_FRAMEBUFFER, s.FboHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glClearColor(v, v, v, v);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glClear(GL_COLOR_BUFFER_BIT);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
}

void Advect(Surface velocity, Surface source, Surface obstacles, Surface dest, float dissipation)
{
	LOG_INFO("Advect", "");
    GLuint p = Programs.Advect;
    glUseProgram(p);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    GLint inverseSize = glGetUniformLocation(p, "InverseSize");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    GLint timeStep = glGetUniformLocation(p, "TimeStep");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    GLint dissLoc = glGetUniformLocation(p, "Dissipation");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    GLint sourceTexture = glGetUniformLocation(p, "SourceTexture");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    GLint obstaclesTexture = glGetUniformLocation(p, "Obstacles");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    glUniform2f(inverseSize, 1.0f / TextureWidth(), 1.0f / TextureHeight());
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform1f(timeStep, TimeStep);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform1f(dissLoc, dissipation);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform1i(sourceTexture, 1);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform1i(obstaclesTexture, 2);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glActiveTexture(GL_TEXTURE0);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindTexture(GL_TEXTURE_2D, velocity.TextureHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glActiveTexture(GL_TEXTURE1);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindTexture(GL_TEXTURE_2D, source.TextureHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glActiveTexture(GL_TEXTURE2);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindTexture(GL_TEXTURE_2D, obstacles.TextureHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    ResetState();
}

void Jacobi(Surface pressure, Surface divergence, Surface obstacles, Surface dest)
{
	LOG_INFO("Jacobi", "");
    GLuint p = Programs.Jacobi;
    glUseProgram(p);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    GLint alpha = glGetUniformLocation(p, "Alpha");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    GLint inverseBeta = glGetUniformLocation(p, "InverseBeta");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    GLint dSampler = glGetUniformLocation(p, "Divergence");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    GLint oSampler = glGetUniformLocation(p, "Obstacles");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    glUniform1f(alpha, -CellSize * CellSize);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform1f(inverseBeta, 0.25f);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform1i(dSampler, 1);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform1i(oSampler, 2);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glActiveTexture(GL_TEXTURE0);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindTexture(GL_TEXTURE_2D, pressure.TextureHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glActiveTexture(GL_TEXTURE1);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindTexture(GL_TEXTURE_2D, divergence.TextureHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glActiveTexture(GL_TEXTURE2);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindTexture(GL_TEXTURE_2D, obstacles.TextureHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    ResetState();
}

void SubtractGradient(Surface velocity, Surface pressure, Surface obstacles, Surface dest)
{
	LOG_INFO("SubtractGradient", "");
    GLuint p = Programs.SubtractGradient;
    glUseProgram(p);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    GLint gradientScale = glGetUniformLocation(p, "GradientScale");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform1f(gradientScale, GradientScale);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    GLint halfCell = glGetUniformLocation(p, "HalfInverseCellSize");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform1f(halfCell, 0.5f / CellSize);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    GLint sampler = glGetUniformLocation(p, "Pressure");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform1i(sampler, 1);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    sampler = glGetUniformLocation(p, "Obstacles");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform1i(sampler, 2);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glActiveTexture(GL_TEXTURE0);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindTexture(GL_TEXTURE_2D, velocity.TextureHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glActiveTexture(GL_TEXTURE1);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindTexture(GL_TEXTURE_2D, pressure.TextureHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glActiveTexture(GL_TEXTURE2);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindTexture(GL_TEXTURE_2D, obstacles.TextureHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    ResetState();
}

void ComputeDivergence(Surface velocity, Surface obstacles, Surface dest)
{
	LOG_INFO("ComputeDivergence", "");
    GLuint p = Programs.ComputeDivergence;
    glUseProgram(p);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    GLint halfCell = glGetUniformLocation(p, "HalfInverseCellSize");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform1f(halfCell, 0.5f / CellSize);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    GLint sampler = glGetUniformLocation(p, "Obstacles");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform1i(sampler, 1);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glActiveTexture(GL_TEXTURE0);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindTexture(GL_TEXTURE_2D, velocity.TextureHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glActiveTexture(GL_TEXTURE1);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindTexture(GL_TEXTURE_2D, obstacles.TextureHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    ResetState();
}

void ApplyImpulse(Surface dest, Vector2 position, float value)
{
	LOG_INFO("ApplyImpulse", "");
    GLuint p = Programs.ApplyImpulse;
    glUseProgram(p);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    GLint pointLoc = glGetUniformLocation(p, "Point");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    GLint radiusLoc = glGetUniformLocation(p, "Radius");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    GLint fillColorLoc = glGetUniformLocation(p, "FillColor");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

//    PRINT_SEPARATOR();
//    LOG_INFO("Position X:", position.X);
//    LOG_INFO("Position Y:", position.Y);
//    LOG_INFO("SplatRadius:", SplatRadius());
//    PRINT_SEPARATOR();

    glUniform2f(pointLoc, (float) position.X, (float) position.Y);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform1f(radiusLoc, SplatRadius());
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform3f(fillColorLoc, value, value, value);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glEnable(GL_BLEND);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    ResetState();
}

void ApplyBuoyancy(Surface velocity, Surface temperature, Surface density, Surface dest)
{
	LOG_INFO("ApplyBuoyancy", "");
    GLuint p = Programs.ApplyBuoyancy;
    glUseProgram(p);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    GLint tempSampler = glGetUniformLocation(p, "Temperature");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    GLint inkSampler = glGetUniformLocation(p, "Density");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    GLint ambTemp = glGetUniformLocation(p, "AmbientTemperature");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    GLint timeStep = glGetUniformLocation(p, "TimeStep");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    GLint sigma = glGetUniformLocation(p, "Sigma");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    GLint kappa = glGetUniformLocation(p, "Kappa");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    glUniform1i(tempSampler, 1);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform1i(inkSampler, 2);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform1f(ambTemp, AmbientTemperature);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform1f(timeStep, TimeStep);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform1f(sigma, SmokeBuoyancy);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform1f(kappa, SmokeWeight);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glActiveTexture(GL_TEXTURE0);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindTexture(GL_TEXTURE_2D, velocity.TextureHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glActiveTexture(GL_TEXTURE1);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindTexture(GL_TEXTURE_2D, temperature.TextureHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glActiveTexture(GL_TEXTURE2);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindTexture(GL_TEXTURE_2D, density.TextureHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    ResetState();
}





const char* FluidInitialize(int width, int height)
{
	ScreenWidth		= width;
	ScreenHeight	= height;

    int tw = TextureWidth();
    int th = TextureHeight();

    PRINT_SEPARATOR();
    LOG_INFO("ScreenWidth:", ScreenWidth);
	LOG_INFO("ScreenHeight:", ScreenHeight);
	LOG_INFO("ViewportWidth:", ViewportWidth);
	LOG_INFO("ViewportHeight:", ViewportHeight);
	PRINT_SEPARATOR();

    Velocity = CreateSlab(tw, th, 2);
    Density = CreateSlab(tw, th, 1);
    Pressure = CreateSlab(tw, th, 1);
    Temperature = CreateSlab(tw, th, 1);
    Divergence = CreateSurface(tw, th, 3);

    InitSlabOps();

    VisualizeProgram = CreateProgram(vs_fluid, 0, fs_visualize);

    Obstacles = CreateSurface(tw, th, 3);
    CreateObstacles(Obstacles, tw, th);

    int w = ViewportWidth;
    int h = ViewportHeight;

    HiresObstacles = CreateSurface(w, h, 1);
    CreateObstacles(HiresObstacles, w, h);

    QuadVao = CreateQuad();
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    ClearSurface(Temperature.Ping, AmbientTemperature);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    return "Fluid";
}

void FluidUpdate(unsigned int elapsedMicroseconds)
{
	glBindVertexArray(QuadVao);

	int tw = TextureWidth();
	int th = TextureHeight();

    glViewport(0, 0, tw, th);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

//    PRINT_SEPARATOR();
//    LOG_INFO("Width:", tw);
//    LOG_INFO("Height:", th);
//    PRINT_SEPARATOR();

	Advect(Velocity.Ping, Velocity.Ping, Obstacles, Velocity.Pong, VelocityDissipation);
    SwapSurfaces(&Velocity);

    Advect(Velocity.Ping, Temperature.Ping, Obstacles, Temperature.Pong, TemperatureDissipation);
    SwapSurfaces(&Temperature);

    Advect(Velocity.Ping, Density.Ping, Obstacles, Density.Pong, DensityDissipation);
    SwapSurfaces(&Density);

    ApplyBuoyancy(Velocity.Ping, Temperature.Ping, Density.Ping, Velocity.Pong);
    SwapSurfaces(&Velocity);

    Vector2 ImpulsePosition = { tw / 2, 0 };

    ApplyImpulse(Temperature.Ping, ImpulsePosition, ImpulseTemperature);
    ApplyImpulse(Density.Ping, ImpulsePosition, ImpulseDensity);

    ComputeDivergence(Velocity.Ping, Obstacles, Divergence);
    ClearSurface(Pressure.Ping, 0);

    for (int i = 0; i < NumJacobiIterations; ++i)
    {
        Jacobi(Pressure.Ping, Divergence, Obstacles, Pressure.Pong);
        SwapSurfaces(&Pressure);
    }

    SubtractGradient(Velocity.Ping, Pressure.Ping, Obstacles, Velocity.Pong);
    SwapSurfaces(&Velocity);

    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
}

#ifdef GRADIENT_COLOR

struct GradientColor
{
	float r, g, b;

	float delta;
	float r_delta;
	float g_delta;
	float b_delta;

	float lower_limit;
	float upper_limit;


	GradientColor() :
		delta(0.02f),
		lower_limit(0.1f),
		upper_limit(1.0f)
	{
		auto rand_color = []()
		{
			return (float)rand()/(float)RAND_MAX;
		};

		r = rand_color();
		g = rand_color();
		b = rand_color();

		InitDeltas();
	}

	void InitDeltas()
	{
		auto rand_delta = [&]()
		{
			return (rand() % 2 == 0 ? 1 : -1) * delta;
		};

	    r_delta = rand_delta();
	    g_delta = rand_delta();
	    b_delta = rand_delta();
	}

	void Update()
	{
		auto gradient = [&](float & v, float & delta)
		{
			v += delta;

			if (v > upper_limit)
			{
				v = upper_limit;
				InitDeltas();
				if (delta > 0.0) delta = -delta;
			}
			else if (v < lower_limit)
			{
				v = lower_limit;
				InitDeltas();
				if (delta < 0.0 ) delta = -delta;
			}
		};

		gradient(r, r_delta);
		gradient(g, g_delta);
		gradient(b, b_delta);
	}

	void Dump()
	{
		LOG_INFO("R:", r);
		LOG_INFO("G:", g);
		LOG_INFO("B:", b);
		LOG_INFO("R-Delta:", r_delta);
		LOG_INFO("G-Delta:", g_delta);
		LOG_INFO("B-Delta:", b_delta);
	}
} g_smoke_color, g_obstacle_color;

#endif //GRADIENT_COLOR

void FluidRender(GLuint windowFbo)
{
    // Bind visualization shader and set up blend state:
    glUseProgram(VisualizeProgram);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    GLint fillColor = glGetUniformLocation(VisualizeProgram, "FillColor");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    GLint scale = glGetUniformLocation(VisualizeProgram, "Scale");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    GLint enableOffset = glGetUniformLocation(VisualizeProgram, "EnableOffset");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    GLint offset = glGetUniformLocation(VisualizeProgram, "Offset");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    GLint viewport = glGetUniformLocation(VisualizeProgram, "Viewport");
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    glEnable(GL_BLEND);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glBindFramebuffer(GL_FRAMEBUFFER, windowFbo);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glClearColor(0, 0, 0, 1);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glClear(GL_COLOR_BUFFER_BIT);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    int w = ViewportWidth;
    int h = ViewportHeight;

    Vector2 AnchorPoint = GetAnchorPoint(ScreenWidth, ScreenHeight, w, h);

    glViewport(AnchorPoint.X, AnchorPoint.Y, w, h);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

#ifdef GRADIENT_COLOR

    static int count = 0;

    if (++count > 6)
    {
    	count = 0;

		g_smoke_color.Update();
		g_obstacle_color.Update();

#if 0
		g_smoke_color.Dump();
		g_obstacle_color.Dump();
#endif
    }

#endif //GRADIENT_COLOR

	glBindVertexArray(QuadVao);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

	float scale_x = 1.0f / (ViewportWidth * 1.0f);
	float scale_y = 1.0f / (ViewportHeight * 1.0f);

    glUniform2f(scale, scale_x, scale_y);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform1i(enableOffset, 1);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform2f(offset, AnchorPoint.X, AnchorPoint.Y);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glUniform2f(viewport, ViewportWidth, ViewportHeight);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    // Draw ink:
    glBindTexture(GL_TEXTURE_2D, Density.Ping.TextureHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
#ifndef GRADIENT_COLOR
    glUniform3f(fillColor, 1, 1, 1);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
#else
    glUniform3f(fillColor, g_smoke_color.r, g_smoke_color.g, g_smoke_color.b);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
#endif //GRADIENT_COLOR
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

	// Draw obstacles:
    glBindTexture(GL_TEXTURE_2D, HiresObstacles.TextureHandle);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
#ifndef GRADIENT_COLOR
    glUniform3f(fillColor, 0.0f, 0.69f, 0.69f);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    //glUniform3f(fillColor, 0.7f, 0.7f, 0.7f);
#else
    glUniform3f(fillColor, g_obstacle_color.r, g_obstacle_color.g, g_obstacle_color.b);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
#endif //GRADIENT_COLOR
    glUniform2f(scale, scale_x, scale_y);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");

    // Disable blending:
    glDisable(GL_BLEND);
    GL_CHECK_CONDITION(GL_NO_ERROR == glGetError(), "GL ERROR");
}

