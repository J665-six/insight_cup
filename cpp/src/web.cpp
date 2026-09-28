#include "insight_cup/web.hpp"

#include <atomic>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <httplib.h>

namespace insight_cup {

namespace {

std::string readFile(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("UI asset not found: " + path.string());
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

void commonHeaders(httplib::Response& response) {
  response.set_header("Cache-Control", "no-store");
  response.set_header("X-Content-Type-Options", "nosniff");
  response.set_header("Referrer-Policy", "no-referrer");
  response.set_header(
      "Content-Security-Policy",
      "default-src 'self'; img-src 'self' data:; style-src 'self'; "
      "script-src 'self'; connect-src 'self'");
}

}  // namespace

class DebugServer::Impl {
 public:
  Impl(RuntimeState& state, Logger& logger) : state_(state), logger_(logger) {
    const fs::path uiRoot =
        fs::path(INSIGHT_CUP_PROJECT_ROOT) / "src/insight_cup/app/ui";
    const std::string index = readFile(uiRoot / "index.html");
    const std::string css = readFile(uiRoot / "app.css");
    const std::string javascript = readFile(uiRoot / "app.js");

    server_.set_keep_alive_max_count(100);
    server_.set_write_timeout(5, 0);
    server_.Get("/", [index](const httplib::Request&, httplib::Response& response) {
      response.set_content(index, "text/html; charset=utf-8");
      commonHeaders(response);
    });
    server_.Get("/app.css",
                [css](const httplib::Request&, httplib::Response& response) {
                  response.set_content(css, "text/css; charset=utf-8");
                  commonHeaders(response);
                });
    server_.Get("/app.js",
                [javascript](const httplib::Request&, httplib::Response& response) {
                  response.set_content(javascript,
                                       "text/javascript; charset=utf-8");
                  commonHeaders(response);
                });
    server_.Get("/api/state",
                [this](const httplib::Request&, httplib::Response& response) {
                  response.set_content(state_.snapshot().dump(),
                                       "application/json; charset=utf-8");
                  commonHeaders(response);
                });
    server_.Get("/api/events",
                [this](const httplib::Request& request,
                       httplib::Response& response) {
                  std::size_t limit = 50;
                  if (request.has_param("limit")) {
                    try {
                      limit = static_cast<std::size_t>(
                          std::stoul(request.get_param_value("limit")));
                    } catch (const std::exception&) {
                      response.status = 400;
                      response.set_content(R"({"error":"invalid limit"})",
                                           "application/json; charset=utf-8");
                      return;
                    }
                  }
                  json payload = {{"events", state_.recentEvents(limit)}};
                  response.set_content(payload.dump(),
                                       "application/json; charset=utf-8");
                  commonHeaders(response);
                });
    server_.Get("/api/frame.jpg",
                [this](const httplib::Request&, httplib::Response& response) {
                  const auto frame = state_.latestFrame();
                  if (frame.empty()) {
                    response.status = 204;
                  } else {
                    response.set_content(
                        std::string(reinterpret_cast<const char*>(frame.data()),
                                    frame.size()),
                        "image/jpeg");
                  }
                  commonHeaders(response);
                });
    server_.Get("/api/stream.mjpg",
                [this](const httplib::Request&, httplib::Response& response) {
                  auto version = std::make_shared<std::uint64_t>(0);
                  response.set_header("Cache-Control", "no-store, max-age=0");
                  response.set_header("Pragma", "no-cache");
                  response.set_chunked_content_provider(
                      "multipart/x-mixed-replace; boundary=insightcupframe",
                      [this, version](std::size_t, httplib::DataSink& sink) {
                        auto [nextVersion, frame, status] = state_.waitForFrame(
                            *version, std::chrono::milliseconds(1000));
                        const bool terminal = status == "completed" ||
                                              status == "stopped" ||
                                              status == "error";
                        if (nextVersion == *version || frame.empty()) {
                          if (terminal) {
                            sink.done();
                            return false;
                          }
                          return true;
                        }
                        *version = nextVersion;
                        std::ostringstream header;
                        header << "--insightcupframe\r\n"
                               << "Content-Type: image/jpeg\r\n"
                               << "Content-Length: " << frame.size()
                               << "\r\n\r\n";
                        const std::string headerText = header.str();
                        if (!sink.write(headerText.data(), headerText.size()) ||
                            !sink.write(
                                reinterpret_cast<const char*>(frame.data()),
                                frame.size()) ||
                            !sink.write("\r\n", 2)) {
                          return false;
                        }
                        if (terminal) {
                          sink.done();
                          return false;
                        }
                        return true;
                      });
                });
    server_.Get("/healthz",
                [this](const httplib::Request&, httplib::Response& response) {
                  json payload = {{"ok", true},
                                  {"status", state_.snapshot()["status"]}};
                  response.set_content(payload.dump(),
                                       "application/json; charset=utf-8");
                  commonHeaders(response);
                });
    server_.Post("/api/control",
                 [this](const httplib::Request& request,
                        httplib::Response& response) {
                   try {
                     const json payload = json::parse(request.body);
                     const auto [accepted, message] =
                         state_.control(payload.at("action").get<std::string>());
                     response.status = accepted ? 200 : 409;
                     response.set_content(
                         json{{"accepted", accepted},
                              {"message", message},
                              {"state", state_.snapshot()}}
                             .dump(),
                         "application/json; charset=utf-8");
                   } catch (const std::exception&) {
                     response.status = 400;
                     response.set_content(R"({"error":"invalid JSON"})",
                                          "application/json; charset=utf-8");
                   }
                   commonHeaders(response);
                 });
    server_.set_error_handler([](const httplib::Request&,
                                 httplib::Response& response) {
      if (response.status == 404) {
        response.set_content(R"({"error":"not found"})",
                             "application/json; charset=utf-8");
      }
      commonHeaders(response);
    });
    server_.set_logger([this](const httplib::Request& request,
                              const httplib::Response& response) {
      logger_.debug("HTTP " + request.method + " " + request.path + " " +
                    std::to_string(response.status));
    });
  }

  RuntimeState& state_;
  Logger& logger_;
  httplib::Server server_;
};

DebugServer::DebugServer(std::string host, int port, RuntimeState& state,
                         Logger& logger)
    : impl_(std::make_unique<Impl>(state, logger)),
      host_(std::move(host)),
      port_(port) {}

DebugServer::~DebugServer() { stop(); }

bool DebugServer::listen() {
  return impl_->server_.listen(host_.c_str(), port_);
}

void DebugServer::stop() {
  if (impl_) {
    impl_->server_.stop();
  }
}

}  // namespace insight_cup
