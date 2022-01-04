/**
 * @brief Messanging system via messenger + listener combo.
 */

#ifndef CAFFE_MESSENGER_H_
#define CAFFE_MESSENGER_H_

#include <map>
#include <list>
#include <string>

#include "caffe/common.hpp"

namespace caffe {

class Listener {
 public:
  virtual ~Listener() {}

  // Handles incoming message.
  virtual void handle(void* message) = 0;
};

class Messenger {
 public:
  typedef std::list<Listener*> ListenersList;
  typedef std::map<string, ListenersList> ListenerRegistry;

  static ListenerRegistry& Registry() {
    static ListenerRegistry* g_registry_ = new ListenerRegistry();
    return *g_registry_;
  }

  // Adds a listener.
  static void AddListener(const string& message_id, Listener* listener) {
    LOG(INFO) << "Adding listener for message " << message_id;
    ListenerRegistry& registry = Registry();
    registry[message_id].push_back(listener);
  }

  // Sends a message to appropriate listeners.
  static void SendMessage(const string& message_id, void* message) {
    DLOG(INFO) << "Sending message " << message_id;
    ListenerRegistry& registry = Registry();
    ListenerRegistry::iterator registry_it = registry.find(message_id);
    if (registry_it != registry.end()) {
      for (ListenersList::iterator listeners_it = registry_it->second.begin();
           listeners_it != registry_it->second.end();
           listeners_it++) {
        (*listeners_it)->handle(message);
      }
    } else {
      DLOG(INFO) << "No listeners found for message " << message_id;
    }
  }

 private:
  // Messenger should never be instantiated.
  Messenger() {}
};

}  // namespace caffe

#endif  // CAFFE_MESSENGER_H_
