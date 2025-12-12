# ML-Based Prediction vs LLM-Based Prediction

## Current Approach: LLM-Based Prediction

This system currently uses **LLM-based prediction** (Google Gemini) rather than traditional **ML-based prediction**. Here's the difference:

### LLM-Based (Current)
- **How it works**: The LLM reads the brief text and generates predictions based on:
  - Pattern recognition from training data (general legal knowledge)
  - In-context learning from historical cases you provide
  - Reasoning about legal arguments
  
- **Pros**:
  - ✅ No training required — works out of the box
  - ✅ Understands complex legal text naturally
  - ✅ Can reason about novel legal arguments
  - ✅ Easy to update prompts/instructions
  
- **Cons**:
  - ❌ Not trained specifically on SCOTUS vote data
  - ❌ Predictions are probabilistic but not calibrated from historical outcomes
  - ❌ Harder to backtest systematically (no ground truth training set)
  - ❌ Can be inconsistent across similar cases

### True ML-Based (Recommended for Production)

For **production-grade ML-based prediction**, you'd want:

1. **Training Data**:
   - Historical SCOTUS cases with:
     - Brief text (petitioner/respondent)
     - Actual votes per Justice
     - Case outcomes
     - Oral argument transcripts
     - Justice questions asked

2. **Feature Engineering**:
   - Extract legal concepts (citations, precedents mentioned)
   - Justice-specific features (past voting patterns, ideology scores)
   - Case characteristics (issue area, circuit split, etc.)
   - Brief quality metrics (length, citation count, etc.)

3. **Model Architecture**:
   - **Option A**: Fine-tune a legal LLM on SCOTUS data
   - **Option B**: Traditional ML (XGBoost, Random Forest) on extracted features
   - **Option C**: Hybrid (LLM for text understanding + ML for vote prediction)

4. **Backtesting Framework**:
   - Train on cases before year X, test on cases after year X
   - Cross-validation with temporal splits
   - Track accuracy per Justice, per issue area
   - Calibrate confidence scores

## Improving This System for ML-Based Prediction

### Short-term (LLM + Better Grounding)
1. **Build a real historical corpus**:
   - Scrape SCOTUS cases from Oyez, Justia, or SCOTUSblog
   - Include actual votes, outcomes, transcripts
   - Store in `data/historical_cases.jsonl`

2. **Better retrieval**:
   - Use embeddings to find truly similar cases
   - Weight predictions by similarity to historical outcomes

3. **Calibration**:
   - Track prediction accuracy over time
   - Adjust confidence scores based on historical performance

### Long-term (True ML)
1. **Data Pipeline**:
   ```python
   # Example: Extract features from briefs
   features = {
       "citation_count": len(re.findall(r'\d+ U\.S\. \d+', brief_text)),
       "precedent_mentions": extract_precedents(brief_text),
       "justice_ideology": get_justice_scores(),
       "case_issue_area": classify_issue(brief_text),
   }
   ```

2. **Training Pipeline**:
   ```python
   # Train on historical data
   X_train = extract_features(historical_briefs)
   y_train = historical_votes  # 9 justices × N cases
   model = train_classifier(X_train, y_train)
   ```

3. **Hybrid Approach** (Best of Both):
   - Use LLM to understand brief text → extract structured features
   - Use ML model to predict votes from features + historical patterns
   - Use LLM to generate questions based on predicted votes

## Backtesting with LLMs

**Current backtesting** compares predicted questions to actual transcript questions (Jaccard similarity). This is useful but limited:

- ✅ Tests if the model generates realistic questions
- ❌ Doesn't test if predictions match actual votes
- ❌ No systematic accuracy metrics

**Better backtesting** would:
1. Use historical cases with known outcomes
2. Predict votes for past cases (temporal split)
3. Compare predicted vs actual votes
4. Track accuracy per Justice, per issue area
5. Measure calibration (are 80% confident predictions right 80% of the time?)

## Recommendations

For **production use**, consider:

1. **Hybrid approach**: LLM for text understanding + ML for vote prediction
2. **Real training data**: Build a corpus of 1000+ SCOTUS cases with votes
3. **Systematic backtesting**: Test on held-out historical cases
4. **Calibration**: Adjust confidence scores based on historical performance
5. **A/B testing**: Compare LLM-only vs ML-only vs hybrid approaches

## Resources

- **SCOTUS Data**: Oyez API, SCOTUSblog, Justia
- **Legal ML**: Papers on predicting court outcomes (e.g., "Predicting Supreme Court Decisions")
- **Fine-tuning**: Hugging Face Transformers for legal domain adaptation

