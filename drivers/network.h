#ifndef NETWORK_H
#define NETWORK_H

#include "KeyMemRT.hpp"
#include "openfhe.h"

using namespace lbcrypto;
using CiphertextT = ConstCiphertext<DCRTPoly>;
using MutableCiphertextT = Ciphertext<DCRTPoly>;
using CCParamsT = CCParams<CryptoContextCKKSRNS>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;

// Network-specific function declarations based on compile-time defines
#ifdef NETWORK_MLP
// MLP function declarations
CiphertextT mlp(CryptoContextT cc, CiphertextT ct);
CryptoContextT mlp__generate_crypto_context();
CryptoContextT mlp__generate_crypto_context(CCParamsT params);
CryptoContextT mlp__configure_crypto_context(CryptoContextT cc, PrivateKeyT sk);
#endif

#ifdef NETWORK_RESNET
// ResNet function declarations
CiphertextT resnet_inference(CryptoContextT cc, CiphertextT ct);
CryptoContextT resnet_inference__generate_crypto_context();
CryptoContextT resnet_inference__generate_crypto_context(CCParamsT params);
CryptoContextT resnet_inference__configure_crypto_context(CryptoContextT cc,
                                                          PrivateKeyT sk);
#endif

#ifdef NETWORK_LOLA
// LolaNet function declarations
CiphertextT lola(CryptoContextT cc, CiphertextT ct);
CryptoContextT lola__generate_crypto_context();
CryptoContextT lola__generate_crypto_context(CCParamsT params);
CryptoContextT lola__configure_crypto_context(CryptoContextT cc,
                                              PrivateKeyT sk);
#endif

#ifdef NETWORK_LENET
// LeNet function declarations
CiphertextT lenet(CryptoContextT cc, CiphertextT ct);
CryptoContextT lenet__generate_crypto_context();
CryptoContextT lenet__generate_crypto_context(CCParamsT params);
CryptoContextT lenet__configure_crypto_context(CryptoContextT cc,
                                               PrivateKeyT sk);
#endif

#ifdef NETWORK_VGG
// VGG function declarations
CiphertextT vgg(CryptoContextT cc, CiphertextT ct);
CryptoContextT vgg__generate_crypto_context();
CryptoContextT vgg__generate_crypto_context(CCParamsT params);
CryptoContextT vgg__configure_crypto_context(CryptoContextT cc, PrivateKeyT sk);
#endif

#ifdef NETWORK_YOLO
// YOLO function declarations
CiphertextT yolo(CryptoContextT cc, CiphertextT ct);
CryptoContextT yolo__generate_crypto_context();
CryptoContextT yolo__generate_crypto_context(CCParamsT params);
CryptoContextT yolo__configure_crypto_context(CryptoContextT cc,
                                              PrivateKeyT sk);
#endif

// Universal interface macros - these get defined based on the network
#ifdef NETWORK_MLP
#define UNIVERSAL_INFERENCE(cc, ct) mlp(cc, ct)
#define UNIVERSAL_CONFIGURE(cc, sk) mlp__configure_crypto_context(cc, sk)
#define UNIVERSAL_GENERATE_CONTEXT() mlp__generate_crypto_context()
#define UNIVERSAL_GENERATE_CONTEXT_WITH_PARAMS(params) mlp__generate_crypto_context(params)
#define NETWORK_NAME "MLP"
#define DEFAULT_DEPTH 30
#define DEFAULT_RING_DIM (1 << 16)
#define INPUT_SIZE 784
#define BOOTSTRAP_ENABLED false
#endif

#ifdef NETWORK_RESNET
#define UNIVERSAL_INFERENCE(cc, ct) resnet_inference(cc, ct)
#define UNIVERSAL_CONFIGURE(cc, sk)                                            \
  resnet_inference__configure_crypto_context(cc, sk)
#define UNIVERSAL_GENERATE_CONTEXT() resnet_inference__generate_crypto_context()
#define UNIVERSAL_GENERATE_CONTEXT_WITH_PARAMS(params) resnet_inference__generate_crypto_context(params)

// Variant-specific network names based on NETWORK_VARIANT_* macros
#if defined(NETWORK_VARIANT_RESNET1)
#define NETWORK_NAME "ResNet1"
#elif defined(NETWORK_VARIANT_RESNET8)
#define NETWORK_NAME "ResNet8"
#elif defined(NETWORK_VARIANT_RESNET10)
#define NETWORK_NAME "ResNet10"
#elif defined(NETWORK_VARIANT_RESNET18)
#define NETWORK_NAME "ResNet18"
#elif defined(NETWORK_VARIANT_RESNET20)
#define NETWORK_NAME "ResNet20"
#elif defined(NETWORK_VARIANT_RESNET32)
#define NETWORK_NAME "ResNet32"
#elif defined(NETWORK_VARIANT_RESNET34)
#define NETWORK_NAME "ResNet34"
#elif defined(NETWORK_VARIANT_RESNET44)
#define NETWORK_NAME "ResNet44"
#elif defined(NETWORK_VARIANT_RESNET50)
#define NETWORK_NAME "ResNet50"
#elif defined(NETWORK_VARIANT_RESNET56)
#define NETWORK_NAME "ResNet56"
#elif defined(NETWORK_VARIANT_RESNET101)
#define NETWORK_NAME "ResNet101"
#elif defined(NETWORK_VARIANT_RESNET110)
#define NETWORK_NAME "ResNet110"
#elif defined(NETWORK_VARIANT_RESNET152)
#define NETWORK_NAME "ResNet152"
#elif defined(NETWORK_VARIANT_RESNET1202)
#define NETWORK_NAME "ResNet1202"
#else
#define NETWORK_NAME "ResNet"  // Fallback for unknown variants
#endif

#define DEFAULT_DEPTH 30
#define DEFAULT_RING_DIM (1 << 16)
#define INPUT_SIZE 3072
#define BOOTSTRAP_ENABLED true
#endif

#ifdef NETWORK_ALEXNET
// Function segment 0 declarations
CiphertextT alexnet_computation(CryptoContextT cc, CiphertextT ct);
CryptoContextT alexnet_computation__generate_crypto_context();
CryptoContextT alexnet_computation__generate_crypto_context(CCParamsT params);
CryptoContextT alexnet_computation__configure_crypto_context(CryptoContextT cc,
                                                             PrivateKeyT sk);

#define UNIVERSAL_INFERENCE(cc, ct) alexnet_computation(cc, ct)
#define UNIVERSAL_CONFIGURE(cc, sk)                                            \
  alexnet_computation__configure_crypto_context(cc, sk)
#define UNIVERSAL_GENERATE_CONTEXT()                                           \
  alexnet_computation__generate_crypto_context()
#define UNIVERSAL_GENERATE_CONTEXT_WITH_PARAMS(params)                         \
  alexnet_computation__generate_crypto_context(params)
#define NETWORK_NAME "AlexNet"
#define DEFAULT_DEPTH 30
#define DEFAULT_RING_DIM (1 << 16)
#define INPUT_SIZE 12288
#define BOOTSTRAP_ENABLED true
#endif

#ifdef NETWORK_LOLA
#define UNIVERSAL_INFERENCE(cc, ct) lola(cc, ct)
#define UNIVERSAL_CONFIGURE(cc, sk) lola__configure_crypto_context(cc, sk)
#define UNIVERSAL_GENERATE_CONTEXT() lola__generate_crypto_context()
#define UNIVERSAL_GENERATE_CONTEXT_WITH_PARAMS(params) lola__generate_crypto_context(params)
#define NETWORK_NAME "Lola"
#define DEFAULT_DEPTH 30
#define DEFAULT_RING_DIM (1 << 16)
#define INPUT_SIZE 1024
#define BOOTSTRAP_ENABLED false
#endif

#ifdef NETWORK_LENET
#define UNIVERSAL_INFERENCE(cc, ct) lenet(cc, ct)
#define UNIVERSAL_CONFIGURE(cc, sk) lenet__configure_crypto_context(cc, sk)
#define UNIVERSAL_GENERATE_CONTEXT() lenet__generate_crypto_context()
#define UNIVERSAL_GENERATE_CONTEXT_WITH_PARAMS(params) lenet__generate_crypto_context(params)
#define NETWORK_NAME "LeNet"
#define DEFAULT_DEPTH 30
#define DEFAULT_RING_DIM (1 << 16)
#define INPUT_SIZE 784 // 28*28 for MNIST
#define BOOTSTRAP_ENABLED false
#endif

#ifdef NETWORK_VGG
#define UNIVERSAL_INFERENCE(cc, ct) vgg(cc, ct)
#define UNIVERSAL_CONFIGURE(cc, sk) vgg__configure_crypto_context(cc, sk)
#define UNIVERSAL_GENERATE_CONTEXT() vgg__generate_crypto_context()
#define UNIVERSAL_GENERATE_CONTEXT_WITH_PARAMS(params) vgg__generate_crypto_context(params)
#define NETWORK_NAME "VGG"
#define DEFAULT_DEPTH 35
#define DEFAULT_RING_DIM (1 << 16)
#define INPUT_SIZE 3072 // 32*32*3 for CIFAR-10
#define BOOTSTRAP_ENABLED true
#endif

#ifdef NETWORK_YOLO
#define UNIVERSAL_INFERENCE(cc, ct) yolo(cc, ct)
#define UNIVERSAL_CONFIGURE(cc, sk) yolo__configure_crypto_context(cc, sk)
#define UNIVERSAL_GENERATE_CONTEXT() yolo__generate_crypto_context()
#define UNIVERSAL_GENERATE_CONTEXT_WITH_PARAMS(params) yolo__generate_crypto_context(params)
#define NETWORK_NAME "YOLO"
#define DEFAULT_DEPTH 45
#define DEFAULT_RING_DIM (1 << 17)
#define INPUT_SIZE 602112 // 448*448*3 for YOLO input
#define BOOTSTRAP_ENABLED true
#endif

#ifndef NETWORK_NAME
#error "No network defined! Use -DNETWORK_<NAME> during compilation"
#endif

#endif // UNIVERSAL_NETWORK_H
