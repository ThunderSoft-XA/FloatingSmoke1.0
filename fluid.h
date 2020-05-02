/*
 * fluid.h
 *
 *  Created on: Sep 6, 2019
 *      Author: little
 */

#ifndef FLUID_H_
#define FLUID_H_


#ifdef _OGLES3
#include "OpenGLES/FrmGLES3.h"
//#include <GLES3/gl3.h>
//#include <GLES2/gl2ext.h>
#else
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#endif



#include <vector>
#include <iostream>



//#include <GL/glew.h>
#include <glm/glm.hpp>
#include <time.h>


//#ifndef NDEBUG
#if 1

#define __DEBUG_PRINT(stream, tag, m, v) \
	do { \
		stream << "[" << tag << "][" << __FILE__ << "@" << __LINE__ << "] " << (m) << " " << (v) << std::endl; \
	} while(0)

#define LOG_INFO(m, v)	__DEBUG_PRINT(g_logfile, "INFO", m, v)
#define LOG_ERROR(m, v)	__DEBUG_PRINT(g_logfile, "ERROR", m, v)

#define PRINT_SEPARATOR() LOG_INFO("----------------------------------------------------", "")

#else

#define LOG_INFO(m, v)
#define LOG_ERROR(m, v)

#define PRINT_SEPARATOR()

#endif //NDEBUG



#ifndef NDEBUG

#define __GL_LOG(get_iv, get_log, obj, v) \
	do { \
		GLsizei len; \
		get_iv(obj, GL_INFO_LOG_LENGTH, &len); \
		LOG_INFO("Length:", len); \
		if (len > 0) \
		{ \
			GLchar *log = new GLchar[len + 1]; \
			get_log(obj, len, &len, log); \
			/*std::cerr << "[ERROR][" << __FILE__ << "@" << __LINE__ << "] " << v << ": " << log << std::endl; */ \
			LOG_ERROR(v, log); \
			delete [] log; \
		} \
	} while(0)

#define SHADER_LOG(_shader, v)		__GL_LOG(glGetShaderiv, glGetShaderInfoLog, _shader, v)

#define PROGRAM_LOG(_program, v)	__GL_LOG(glGetProgramiv, glGetProgramInfoLog, _program, v)

#else

#define SHADER_LOG(_shader, v)
#define PROGRAM_LOG(_program, v)

#endif //_DEBUG


#define GL_CHECK_CONDITION(condition, msg) \
	do { \
		if (!(condition)) { \
			LOG_ERROR("[GL ERROR]", msg); \
		} \
	} while(0)

extern const GLchar *vs_fluid;
extern const GLchar *fs_fill;
extern const GLchar *fs_advect;
extern const GLchar *fs_jacobi;
extern const GLchar *fs_subtract_gradient;
extern const GLchar *fs_compute_divergence;
extern const GLchar *fs_splat;
extern const GLchar *fs_buoyancy;
extern const GLchar *fs_visualize;



const char* FluidInitialize(int width, int height);
void FluidUpdate(unsigned int elapsedMicroseconds);
void FluidRender(GLuint windowFbo);


#define LOG_FILE "/sdcard/smok.log"



#endif /* FLUID_H_ */