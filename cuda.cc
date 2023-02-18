#include <iostream>
#include "all.h"

void foo_cuda() {
  std::cout << "FOO CUDA" << std::endl;
}
void foo2_cuda() {
  std::cout << "FOO2 CUDA" << std::endl;
}

OP_IMPL(foo, CUDA) {
  Dispatcher::GetSingleton().Register("foo", DispatchKey::CUDA, reinterpret_cast<void*>(&foo_cuda));
}

OP_IMPL(foo2, CUDA) {
  Dispatcher::GetSingleton().Register("foo2", DispatchKey::CUDA, reinterpret_cast<void*>(&foo2_cuda));
}