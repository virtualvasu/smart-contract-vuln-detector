
# Smart Contract Vulnerability Detection Benchmark Report

Generated on: 2025-11-11 01:51:53

## Executive Summary

This report compares the performance of our CodeBERT-based vulnerability detection model against traditional static analysis tools (Slither and Mythril) on a test set of Solidity smart contract functions.

### Dataset Overview
- Total test functions analyzed: 554
- Ground truth vulnerability rate: 7.2%
- Vulnerability categories covered: 7 types

## Tool Performance Comparison


### Performance Metrics

                tool  n_predictions  accuracy  precision  recall    f1  true_positives  false_positives  true_negatives  false_negatives
Our Model (CodeBERT)            554     0.942        0.7    0.35 0.467              14                6             508               26
             Slither            554     0.928        0.0    0.00 0.000               0                0             514               40
             Mythril            554     0.928        0.0    0.00 0.000               0                0             514               40


### Key Findings

- Best F1 Score: Our Model (CodeBERT) with 0.467
- Highest Precision: Our Model (CodeBERT) with 0.700
- Highest Recall: Our Model (CodeBERT) with 0.350
- Our Model (CodeBERT) Coverage: 100.0% (554/554 functions)
- Slither Coverage: 100.0% (554/554 functions)
- Mythril Coverage: 100.0% (554/554 functions)

### Tool Agreement Analysis

- Our Model vs Slither: 96.4% agreement (554 overlapping predictions)
- Our Model vs Mythril: 96.4% agreement (554 overlapping predictions)
- Slither vs Mythril: 100.0% agreement (554 overlapping predictions)


## Methodology Notes

1. **Static Tool Analysis**: Slither and Mythril were run on contract files with a timeout of 60-120 seconds per contract.
2. **Function-Level Mapping**: Tool outputs were mapped to function-level predictions using heuristic approaches.
3. **Evaluation Metrics**: Standard classification metrics (Accuracy, Precision, Recall, F1) were computed.
4. **Limitations**: 
   - Limited sample size for static tool analysis due to computational constraints
   - Function-level granularity mapping may introduce noise
   - Static tools may detect different vulnerability types than our training data

## Conclusions

This benchmark provides insights into the relative strengths and weaknesses of different vulnerability detection approaches. Machine learning models like our CodeBERT implementation may offer advantages in terms of consistency and scalability, while static analysis tools provide rule-based detection with different coverage patterns.

For production use, a hybrid approach combining multiple detection methods may yield the best results.
