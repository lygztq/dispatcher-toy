#include "all.h"


OP_DEF(foo) {
  Dispatcher::GetSingleton().DefineOperator("foo");
}

OP_DEF(foo2) {
  Dispatcher::GetSingleton().DefineOperator("foo2");
}

int main() {
  Dispatcher::GetSingleton().DispatchCall<void>("foo", DispatchKey::CPU);
  Dispatcher::GetSingleton().DispatchCall<void>("foo", DispatchKey::CUDA);
  Dispatcher::GetSingleton().DispatchCall<void>("foo2", DispatchKey::CPU);
  Dispatcher::GetSingleton().DispatchCall<void>("foo2", DispatchKey::CUDA);
  return 0;
}