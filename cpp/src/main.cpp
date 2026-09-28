#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "insight_cup/config.hpp"
#include "insight_cup/logging.hpp"
#include "insight_cup/runtime.hpp"
#include "insight_cup/state.hpp"
#include "insight_cup/web.hpp"

namespace {

std::atomic<bool> interrupted{false};

void handleSignal(int) { interrupted.store(true); }

}  // namespace

int main(int argc, char** argv) {
  using namespace insight_cup;
  try {
    ParsedCommandLine commandLine = parseCommandLine(argc, argv);
    if (commandLine.showHelp) {
      std::cout << commandLineHelp();
      return 0;
    }
    RuntimeConfig config = std::move(commandLine.config);
    Logger logger(config.verbose);
    const fs::path sessionDirectory =
        createSessionDirectory(config.outputRoot, config.sessionId);
    RuntimeState state(!config.noUi);
    state.configureSession(config.source, sessionDirectory.filename().string(),
                           sessionDirectory);
    EventJournal journal(sessionDirectory / "events.jsonl", logger);
    RecognitionRuntime runtime(config, sessionDirectory, state, journal, logger);

    logger.info("SESSION id=" + sessionDirectory.filename().string() +
                " source=" + config.source);
    logger.info("OUTPUT path=" + sessionDirectory.string());

    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);
    std::atomic<bool> finished{false};
    std::atomic<DebugServer*> serverPointer{nullptr};
    std::thread signalWatcher([&] {
      while (!finished.load()) {
        if (interrupted.load()) {
          logger.info("STOP reason=signal");
          state.control("stop");
          if (DebugServer* server = serverPointer.load()) {
            server->stop();
          }
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    });

    if (config.noUi) {
      runtime.run();
      finished.store(true);
      signalWatcher.join();
      return state.snapshot()["status"] == "error" ? 1 : 0;
    }

    DebugServer server(config.host, config.port, state, logger);
    serverPointer.store(&server);
    const std::string displayHost =
        config.host == "0.0.0.0" || config.host == "::" ? "127.0.0.1"
                                                          : config.host;
    const std::string url = "http://" + displayHost + ":" +
                            std::to_string(config.port);
    logger.info("UI url=" + url);
    std::thread runtimeThread([&] { runtime.run(); });
    std::thread completionWatcher;
    if (config.exitOnComplete) {
      completionWatcher = std::thread([&] {
        runtimeThread.join();
        server.stop();
      });
    }
    if (config.openBrowser) {
      std::thread([url] {
        std::this_thread::sleep_for(std::chrono::milliseconds(800));
        const std::string command = "xdg-open '" + url +
                                    "' >/dev/null 2>&1";
        const int result = std::system(command.c_str());
        (void)result;
      }).detach();
    }
    if (!server.listen()) {
      logger.error("Could not listen on " + config.host + ":" +
                   std::to_string(config.port));
      state.control("stop");
    }
    if (!config.exitOnComplete && runtimeThread.joinable()) {
      state.control("stop");
      runtimeThread.join();
    }
    if (completionWatcher.joinable()) {
      completionWatcher.join();
    }
    finished.store(true);
    signalWatcher.join();
    return state.snapshot()["status"] == "error" ? 1 : 0;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << std::endl;
    return 2;
  }
}
