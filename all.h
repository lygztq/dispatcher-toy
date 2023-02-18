#ifndef ALL_H_
#define ALL_H_
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <type_traits>
#include <memory>

enum class DispatchKey {
  CPU = 0,
  CUDA = 1,
  ROCM = 2,
  VULKUN = 3,
  NONE = 31
};

template <typename IndexTy>
constexpr auto DispatchKeyToIndex(DispatchKey key) ->
std::enable_if_t<std::is_integral_v<IndexTy>, IndexTy> {
  return static_cast<IndexTy>(key);
}

inline const char* DispatchKeyName(DispatchKey key) {
  switch (key) {
    case DispatchKey::CPU:
      return "CPU";
    case DispatchKey::CUDA:
      return "CUDA";
    case DispatchKey::ROCM:
      return "ROCM";
    case DispatchKey::VULKUN:
      return "VULKUN";
    default:
      return "NONE";
  }
}

class OperatorHandle {
public:
  OperatorHandle() = default;
  OperatorHandle(const std::string& name) : name_(name) {
    kernel_tbl_[DispatchKey::NONE] = nullptr; // fallback mechanism
  }

  bool HasDefinition(DispatchKey key) const {
    if (key == DispatchKey::NONE)
      return kernel_tbl_.at(key) != nullptr;
    return kernel_tbl_.find(key) != kernel_tbl_.end();
  }

  void Register(DispatchKey key, void* func) {
    if (HasDefinition(key)) {
      if (kernel_tbl_[key] != func) {
        throw std::runtime_error(std::string("Find multiple definition of operator ") + name_ + std::string(" with dispatch type ") + std::string(DispatchKeyName(key)));
      }
    } else {
      kernel_tbl_[key] = func;
    }
  }

  const void* Get(DispatchKey key) const {
    if (HasDefinition(key)) {
      return kernel_tbl_.at(key);
    } else {
      return kernel_tbl_.at(DispatchKey::NONE);
    }
  }
private:
  std::string name_;
  std::unordered_map<DispatchKey, void*> kernel_tbl_;
};

template <typename Ty>
class Singleton {
public:
  static Ty& GetSingleton() {
    static Ty singleton_obj;
    return singleton_obj;
  }
};

class Dispatcher final : public Singleton<Dispatcher> {
public:
  void DefineOperator(const std::string& name) {
    if (!HasDefinition(name)) {
      // throw std::runtime_error("Found multiple definition of operator " + name);
      op_tbl_[name] = std::make_unique<OperatorHandle>(name);
    }
  }

  template <typename Return, typename ...Args>
  std::enable_if_t<!std::is_void_v<Return>, Return> DispatchCall(const std::string& name, DispatchKey key, Args&&... args) const {
    if (!HasDefinition(name)) { // TODO add fallback
      throw std::runtime_error(std::string("Cannot find operator ") + name);
    }
    auto* handle = op_tbl_.at(name).get();
    void* raw_kernel_fn = handle->Get(key);
    if (raw_kernel_fn == nullptr) {
      throw std::runtime_error(std::string("Cannot find kernel with dispatch key ") + std::string(DispatchKeyName(key)) + std::string(" of operator ") + name);
    }
    using typed_fn_t = Return(*)(Args...);
    return reinterpret_cast<typed_fn_t>(raw_kernel_fn)(std::forward<Args>(args)...);
  }

  template <typename Return, typename ...Args>
  std::enable_if_t<std::is_void_v<Return>, void> DispatchCall(const std::string& name, DispatchKey key, Args&&... args) const {
    if (!HasDefinition(name)) { // TODO add fallback
      throw std::runtime_error(std::string("Cannot find operator ") + name);
    }
    auto* handle = op_tbl_.at(name).get();
    const void* raw_kernel_fn = handle->Get(key);
    if (raw_kernel_fn == nullptr) {
      throw std::runtime_error(std::string("Cannot find kernel with dispatch key ") + std::string(DispatchKeyName(key)) + std::string(" of operator ") + name);
    }
    using typed_fn_t = Return(*)(Args...);
    reinterpret_cast<typed_fn_t>(raw_kernel_fn)(std::forward<Args>(args)...);
  }

  bool HasDefinition(const std::string& name) const {
    return op_tbl_.find(name) != op_tbl_.end();
  }

  void Register(const std::string& name, DispatchKey key, void* fn) {
    if (!HasDefinition(name)) {
      DefineOperator(name);
    }
    op_tbl_.at(name)->Register(key, fn);
  }

private:
  std::unordered_map<std::string, std::unique_ptr<OperatorHandle>> op_tbl_;
};

class StaticRegister {
  using static_reg_fn_t = void(*)(void);
public:
  StaticRegister(static_reg_fn_t init_fn) {
    init_fn();
  }
};

#define CONCAT(a, b) a##b

#ifdef __COUNTER__
#define OP_DEF_UID __COUNTER__
// #define C10_ANONYMOUS_VARIABLE(str) C10_CONCATENATE(str, __COUNTER__)
#else
#define OP_DEF_UID __LINE__
// #define C10_ANONYMOUS_VARIABLE(str) C10_CONCATENATE(str, __LINE__)
#endif

#define OP_DEF(name) _OP_DEF(name, OP_DEF_UID)

#define _OP_DEF(name, uid) \
  static void CONCAT(OPERATOR_DEF_INIT_##name, uid)(); \
  static StaticRegister CONCAT(OPERATOR_STATIC_DEF_INIT_##name, uid)(&CONCAT(OPERATOR_DEF_INIT_##name, uid)); \
  void CONCAT(OPERATOR_DEF_INIT_##name, uid)()

#define OP_IMPL(name, key) _OP_IMPL(name, key, OP_DEF_UID)

#define _OP_IMPL(name, key, uid) \
  static void CONCAT(OPERATOR_IMPL_INIT_##name##key, uid)(); \
  static StaticRegister CONCAT(OPERATOR_STATIC_IMPL_INIT_##name##key, uid)(&CONCAT(OPERATOR_IMPL_INIT_##name##key, uid)); \
  void CONCAT(OPERATOR_IMPL_INIT_##name##key, uid)()
#endif // ALL_H_