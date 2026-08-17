#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>
#include <GL/glew.h>

#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#include <thread>

#include "viewer_cuda.h"

typedef unsigned char uchar;

std::mutex mtx;

// ---- 现代 OpenGL shader（core profile，兼容 WSLg 的 D3D12 转发 OpenGL）----
static const char* VERTEX_SHADER = R"(
#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 color;
uniform mat4 mvp;
out vec3 vcolor;
void main() {
  gl_Position = mvp * vec4(position, 1.0);
  vcolor = color;
}
)";

static const char* FRAGMENT_SHADER = R"(
#version 330 core
in vec3 vcolor;
out vec4 fragColor;
void main() {
  fragColor = vec4(vcolor, 1.0);
}
)";

static GLuint compile_shader(GLenum type, const char* src) {
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok = 0;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[1024];
    glGetShaderInfoLog(s, 1024, nullptr, log);
    std::cerr << "[viewer] shader compile error: " << log << std::endl;
  }
  return s;
}

static GLuint create_program() {
  GLuint vs = compile_shader(GL_VERTEX_SHADER, VERTEX_SHADER);
  GLuint fs = compile_shader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
  GLuint p = glCreateProgram();
  glAttachShader(p, vs);
  glAttachShader(p, fs);
  glLinkProgram(p);
  GLint ok = 0;
  glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[1024];
    glGetProgramInfoLog(p, 1024, nullptr, log);
    std::cerr << "[viewer] program link error: " << log << std::endl;
  }
  glDeleteShader(vs);
  glDeleteShader(fs);
  return p;
}


class Viewer {
  public:
    Viewer(
      const torch::Tensor image,
      const torch::Tensor poses,
      const torch::Tensor points,
      const torch::Tensor colors,
      const torch::Tensor intrinsics);

    void close() {
      running = false;
    };

    void join() {
      tViewer.join();
    };

    void update_image(torch::Tensor img) {
      mtx.lock();
      redraw = true;
      image = img.permute({1,2,0}).to(torch::kCPU);
      mtx.unlock();
    }

    // main visualization
    void run();

  private:
    bool running;
    std::thread tViewer;

    int w;
    int h;
    int ux;

    int nPoints, nFrames;

    torch::Tensor image;
    torch::Tensor poses;
    torch::Tensor points;
    torch::Tensor colors;
    torch::Tensor intrinsics;

    bool redraw;

    torch::Tensor transformMatrix;
    void drawPoints(const pangolin::OpenGlMatrix& mvp);
    void drawPoses(const pangolin::OpenGlMatrix& mvp);

    void initGL();
    void destroyGL();

    // 现代 OpenGL 资源
    GLuint vbo, cbo;        // 点云的 vertex / color buffer
    GLuint points_vao;
    GLuint traj_vao, traj_vbo;  // 轨迹线
    GLuint prog;
    GLint mvp_loc;
};


Viewer::Viewer(
      const torch::Tensor image,
      const torch::Tensor poses, 
      const torch::Tensor points,
      const torch::Tensor colors,
      const torch::Tensor intrinsics)
  : image(image), poses(poses), points(points), colors(colors), intrinsics(intrinsics)
{
  running = true;
  redraw = true;
  nFrames = poses.size(0);
  nPoints = points.size(0);

  ux = 0;
  h = image.size(0);
  w = image.size(1);

  tViewer = std::thread(&Viewer::run, this);
};


void Viewer::drawPoints(const pangolin::OpenGlMatrix& mvp) {
  if (points.size(0) == 0) return;

  torch::Tensor xyz_cpu = points.to(torch::kCPU).contiguous();
  torch::Tensor rgb_cpu = colors.to(torch::kCPU).contiguous();

  glUseProgram(prog);
  float mvp_f[16];
  for (int i = 0; i < 16; i++) mvp_f[i] = (float)mvp.m[i];
  glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, mvp_f);

  glBindVertexArray(points_vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, 3 * points.size(0) * sizeof(float),
               xyz_cpu.data_ptr<float>(), GL_DYNAMIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, cbo);
  glBufferData(GL_ARRAY_BUFFER, 3 * points.size(0) * sizeof(uchar),
               rgb_cpu.data_ptr<uchar>(), GL_DYNAMIC_DRAW);

  glDrawArrays(GL_POINTS, 0, points.size(0));
  glBindVertexArray(0);
}


void Viewer::drawPoses(const pangolin::OpenGlMatrix& mvp) {
  if (nFrames < 2) return;

  // transformMatrix: [nFrames, 4, 4] camera-to-world（run 里已 transpose + cpu）
  float* tptr = transformMatrix.data_ptr<float>();

  std::vector<float> traj;
  traj.reserve(nFrames * 3);
  for (int i = 0; i < nFrames; i++) {
    // 平移 = 第 4 列前 3 个元素（column-major: idx 12,13,14）
    traj.push_back(tptr[16 * i + 12]);
    traj.push_back(tptr[16 * i + 13]);
    traj.push_back(tptr[16 * i + 14]);
  }

  glUseProgram(prog);
  float mvp_f[16];
  for (int i = 0; i < 16; i++) mvp_f[i] = (float)mvp.m[i];
  glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, mvp_f);

  // 轨迹线用固定红色（禁用 color array，用常量）
  glBindVertexArray(traj_vao);
  glDisableVertexAttribArray(1);
  glVertexAttrib3f(1, 1.0f, 0.0f, 0.0f);

  glBindBuffer(GL_ARRAY_BUFFER, traj_vbo);
  glBufferData(GL_ARRAY_BUFFER, traj.size() * sizeof(float),
               traj.data(), GL_DYNAMIC_DRAW);

  glDrawArrays(GL_LINE_STRIP, 0, nFrames);
  glBindVertexArray(0);
}


void Viewer::initGL() {
  prog = create_program();
  mvp_loc = glGetUniformLocation(prog, "mvp");

  // 点云 VAO（位置 + 颜色）
  glGenVertexArrays(1, &points_vao);
  glGenBuffers(1, &vbo);
  glGenBuffers(1, &cbo);
  glBindVertexArray(points_vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
  glBindBuffer(GL_ARRAY_BUFFER, cbo);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_UNSIGNED_BYTE, GL_TRUE, 0, (void*)0);
  glBindVertexArray(0);

  // 轨迹 VAO（只有位置）
  glGenVertexArrays(1, &traj_vao);
  glGenBuffers(1, &traj_vbo);
  glBindVertexArray(traj_vao);
  glBindBuffer(GL_ARRAY_BUFFER, traj_vbo);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
  glBindVertexArray(0);
}


void Viewer::destroyGL() {
  glDeleteBuffers(1, &vbo);
  glDeleteBuffers(1, &cbo);
  glDeleteBuffers(1, &traj_vbo);
  glDeleteVertexArrays(1, &points_vao);
  glDeleteVertexArrays(1, &traj_vao);
  glDeleteProgram(prog);
}


void Viewer::run() {

  // initialize OpenGL buffers
  pangolin::CreateWindowAndBind("DPVO", 2*640, 2*480);

  const int UI_WIDTH = 180;
  glEnable(GL_DEPTH_TEST);

  pangolin::OpenGlRenderState Visualization3D_camera(
    pangolin::ProjectionMatrix(w, h, 400, 400, w/2, h/2, 0.1, 500),
    pangolin::ModelViewLookAt(-0, -1, -1, 0, 0, 0, pangolin::AxisNegY));

  pangolin::View& Visualization3D_display = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -w/(float)h);

  initGL();
  std::cerr << "RUN: after initGL" << std::endl;

  while( !pangolin::ShouldQuit() && running ) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    Visualization3D_display.Activate(Visualization3D_camera);
    std::cerr << "D1 after Activate" << std::endl;

    // 计算 MVP（投影 * 模型视图）
    pangolin::OpenGlMatrix mvp =
        Visualization3D_camera.GetProjectionMatrix() *
        Visualization3D_camera.GetModelViewMatrix();
    std::cerr << "D2 after mvp" << std::endl;

    // pose 矩阵转 CPU（用于轨迹线）
    transformMatrix = poseToMatrix(poses);
    transformMatrix = transformMatrix.transpose(1,2);
    transformMatrix = transformMatrix.contiguous().to(torch::kCPU);
    std::cerr << "D3 after poseToMatrix" << std::endl;

    drawPoints(mvp);
    std::cerr << "D4 after drawPoints" << std::endl;
    drawPoses(mvp);
    std::cerr << "D5 after drawPoses" << std::endl;
    // 解绑 shader 和 VAO，避免和 Pangolin 内部的固定管线渲染冲突
    glUseProgram(0);
    glBindVertexArray(0);
    std::cerr << "D6 after unbind" << std::endl;

    pangolin::FinishFrame();
    std::cerr << "D7 after FinishFrame" << std::endl;
  }

  destroyGL();
  running = false;

  exit(1);
}


namespace py = pybind11;

PYBIND11_MODULE(dpviewerx, m) {
  py::class_<Viewer>(m, "Viewer")
    .def(py::init<const torch::Tensor,
                  const torch::Tensor,
                  const torch::Tensor,
                  const torch::Tensor,
                  const torch::Tensor>())
    .def("update_image", &Viewer::update_image)
    .def("join", &Viewer::join)
    .def("close", &Viewer::close);
}
