#ifndef KEY_COMPRESSION_H_
#define KEY_COMPRESSION_H_

#include "Logger.hpp"
#include "openfhe.h"

using namespace lbcrypto;

class RNSKeyCompressor {
public:
  /**
   * Compresses a single evaluation key to match a specific ciphertext level
   * @param evalKey Reference to the evaluation key to compress
   * @param targetCiphertext Ciphertext whose level the key should match
   * @return true if compression succeeded
   */
  static bool CompressKeyToLevel(EvalKey<DCRTPoly> &evalKey,
                                 ConstCiphertext<DCRTPoly> targetCiphertext) {
    if (!evalKey)
      return false;

    // Get target Q tower count from ciphertext
    size_t targetQTowers = GetCiphertextTowerCount(targetCiphertext);

    // Get P tower count from crypto parameters
    auto cryptoParams = targetCiphertext->GetCryptoParameters();
    auto rnsCryptoParams =
        std::dynamic_pointer_cast<CryptoParametersRNS>(cryptoParams);
    if (!rnsCryptoParams)
      return false;

    auto paramsP = rnsCryptoParams->GetParamsP();
    size_t pTowers = paramsP ? paramsP->GetParams().size() : 0;
    // Get current key structure
    auto aVector = evalKey->GetAVector();
    if (aVector.empty())
      return false;

    size_t currentTotalTowers = aVector[0].GetParams()->GetParams().size();
    size_t currentQTowers = currentTotalTowers - pTowers;

    if (currentQTowers <= targetQTowers) {
      // Set Q size for existing keys (even if no compression needed)
      evalKey->SetDynamicQSize(currentQTowers);
      return true;
    }

    // Perform Q tower compression while preserving P towers
    if (!CompressKeyStructure(evalKey, targetQTowers, pTowers)) {
      LOG_DEBUG("Failed to compress key structure for targetQTowers {}",
                targetQTowers);
      return false;
    }

    // CRITICAL: Set the dynamic Q size for the compressed key
    evalKey->SetDynamicQSize(targetQTowers);
    return true;
  }

  /**
   * Compresses a single evaluation key to a specific target level
   * @param evalKey Reference to the evaluation key to compress
   * @param targetQTowers Target number of Q towers to retain
   * @param cryptoParams Crypto parameters to extract P tower count
   * @return true if compression succeeded
   */
  static bool CompressKeyToTargetLevel(
      EvalKey<DCRTPoly> &evalKey, size_t targetQTowers,
      const std::shared_ptr<CryptoParametersBase<DCRTPoly>> cryptoParams) {
    if (!evalKey)
      return false;

    auto rnsCryptoParams =
        std::dynamic_pointer_cast<CryptoParametersRNS>(cryptoParams);
    if (!rnsCryptoParams)
      return false;

    auto paramsP = rnsCryptoParams->GetParamsP();
    size_t pTowers = paramsP ? paramsP->GetParams().size() : 0;

    auto aVector = evalKey->GetAVector();
    if (aVector.empty())
      return false;

    size_t currentTotalTowers = aVector[0].GetParams()->GetParams().size();
    size_t currentQTowers = currentTotalTowers - pTowers;

    if (currentQTowers <= targetQTowers) {
      evalKey->SetDynamicQSize(currentQTowers);
      return true;
    }

    if (!CompressKeyStructure(evalKey, targetQTowers, pTowers)) {
      return false;
    }

    evalKey->SetDynamicQSize(targetQTowers);
    return true;
  }

  // Fixes the evaluation key's SizeQ parameter since ser+deser loses that
  // information
  static bool RestoreDynamicQSize(
      EvalKey<DCRTPoly> &evalKey,
      const std::shared_ptr<CryptoParametersBase<DCRTPoly>> cryptoParams) {
    if (!evalKey)
      return false;

    auto rnsCryptoParams =
        std::dynamic_pointer_cast<CryptoParametersRNS>(cryptoParams);
    if (!rnsCryptoParams)
      return false;

    auto paramsP = rnsCryptoParams->GetParamsP();
    size_t pTowers = paramsP ? paramsP->GetParams().size() : 0;

    auto aVector = evalKey->GetAVector();
    if (aVector.empty())
      return false;

    // Calculate actual Q towers from the deserialized key structure
    size_t totalTowers = aVector[0].GetParams()->GetParams().size();
    size_t actualQTowers = totalTowers - pTowers;

    // Restore the DynamicQSize to match the compressed state
    evalKey->SetDynamicQSize(actualQTowers);

    return true;
  }

  /**
   * Utility function to compress multiple keys using the single-key function
   */
  static bool
  CompressKeysToLevel(std::map<usint, EvalKey<DCRTPoly>> &evalKeyMap,
                      ConstCiphertext<DCRTPoly> targetCiphertext) {
    bool allSuccess = true;

    for (auto &keyPair : evalKeyMap) {
      if (!CompressKeyToLevel(keyPair.second, targetCiphertext)) {
        allSuccess = false;
      }
    }

    return allSuccess;
  }

  /**
   * Utility function to compress multiple keys to target level
   */
  static bool CompressKeysToTargetLevel(
      std::map<usint, EvalKey<DCRTPoly>> &evalKeyMap, size_t targetQTowers,
      const std::shared_ptr<CryptoParametersBase<DCRTPoly>> cryptoParams) {
    bool allSuccess = true;

    for (auto &keyPair : evalKeyMap) {
      if (!CompressKeyToTargetLevel(keyPair.second, targetQTowers,
                                    cryptoParams)) {
        allSuccess = false;
      }
    }

    return allSuccess;
  }

  static size_t GetCiphertextTowerCount(ConstCiphertext<DCRTPoly> ciphertext) {
    const auto &elements = ciphertext->GetElements();
    if (elements.empty())
      return 0;
    return elements[0].GetParams()->GetParams().size();
  }

private:
  /**
   * Compresses a single evaluation key's structure
   */
  static bool CompressKeyStructure(EvalKey<DCRTPoly> &evalKey,
                                   size_t targetQTowers, size_t pTowers) {

    auto aVector = evalKey->GetAVector();
    auto bVector = evalKey->GetBVector();

    if (aVector.empty() || bVector.empty())
      return false;

    // Get current parameters
    const auto &originalParams = aVector[0].GetParams();
    const auto &originalParamVec = originalParams->GetParams();

    size_t currentTotalTowers = originalParamVec.size();
    size_t currentQTowers = currentTotalTowers - pTowers;

    if (targetQTowers >= currentQTowers)
      return true; // No compression needed

    // Create new parameter vector: first targetQTowers Q + all P towers
    std::vector<std::shared_ptr<ILNativeParams>> newParamVec;
    newParamVec.reserve(targetQTowers + pTowers);

    // Add target Q parameters (first targetQTowers)
    for (size_t i = 0; i < targetQTowers; ++i) {
      newParamVec.push_back(originalParamVec[i]);
    }

    // Add all P parameters (they come after Q parameters)
    size_t pStartIndex = currentQTowers;
    for (size_t i = 0; i < pTowers; ++i) {
      newParamVec.push_back(originalParamVec[pStartIndex + i]);
    }

    // Create new parameter object
    auto newParams = std::make_shared<ILDCRTParams<DCRTPoly::Integer>>(
        originalParams->GetCyclotomicOrder(), newParamVec);

    // Compress each polynomial in the vectors
    for (size_t k = 0; k < aVector.size(); ++k) {
      aVector[k] = CompressPolynomial(aVector[k], newParams, targetQTowers,
                                      currentQTowers, pTowers);
      bVector[k] = CompressPolynomial(bVector[k], newParams, targetQTowers,
                                      currentQTowers, pTowers);
    }

    // Update the evaluation key
    evalKey->ClearKeys();
    evalKey->SetAVector(std::move(aVector));
    evalKey->SetBVector(std::move(bVector));

    return true;
  }

  /**
   * Compresses a single DCRTPoly to new Q||P structure
   */
  static DCRTPoly
  CompressPolynomial(const DCRTPoly &originalPoly,
                     std::shared_ptr<ILDCRTParams<DCRTPoly::Integer>> newParams,
                     size_t targetQTowers, size_t currentQTowers,
                     size_t pTowers) {

    // Create new polynomial with compressed parameter set
    DCRTPoly newPoly(newParams, originalPoly.GetFormat(), true);

    const auto &originalElements = originalPoly.GetAllElements();
    auto &newElements = newPoly.GetAllElements();

    // Copy first targetQTowers Q elements
    for (size_t i = 0; i < targetQTowers; ++i) {
      newElements[i] = originalElements[i];
    }

    // Copy all P elements (they start after current Q towers in original)
    size_t pStartInOriginal = currentQTowers;
    size_t pStartInNew = targetQTowers;
    for (size_t i = 0; i < pTowers; ++i) {
      newElements[pStartInNew + i] = originalElements[pStartInOriginal + i];
    }

    return newPoly;
  }
};

#endif
