#include <iostream>
#include "all.h"

void foo_cpu() {
  std::cout << "FOO CPU" << std::endl;
}
void foo2_cpu() {
  std::cout << "FOO2 CPU" << std::endl;
}

OP_IMPL(foo, CPU) {
  Dispatcher::GetSingleton().Register("foo", DispatchKey::CPU, reinterpret_cast<void*>(&foo_cpu));
}

OP_IMPL(foo2, CPU) {
  Dispatcher::GetSingleton().Register("foo2", DispatchKey::CPU, reinterpret_cast<void*>(&foo2_cpu));
}