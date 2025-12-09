<<<<<<< HEAD
# SafeStride Project

SafeStride is a project designed to process Inertial Measurement Unit (IMU) data to predict Ground Reaction Forces (GRF) that are ready for integration with OpenSim for inverse kinematics (IK) and inverse dynamics (ID) analysis.

## Quickstart

### Setting up the Environment

1. **Create a Conda Environment**
   ```bash
   conda create -n safestride python=3.10
   conda activate safestride
   ```

2. **Install Dependencies**
   ```bash
   pip install numpy pandas scipy scikit-learn matplotlib joblib pytest ezc3d
   ```

### Generating Synthetic Data

Run the following script to generate synthetic IMU and GRF data:
```bash
python src/synthetic.py
```

### Training a Model

Train a model using the synthetic data:
```bash
python src/train.py --imu_csv data/sample/imu_trial01.csv --grf_path data/sample/grf_trial01.csv --bw_kg 70
```

### Predicting GRF

Predict GRF using a trained model:
```bash
python src/predict_grf.py --imu_csv data/sample/imu_trial01.csv --model_pkl out/model.pkl --bw_kg 70
```

### Evaluating Predictions

Evaluate the predicted GRF against true data:
```bash
python src/eval_compare.py --true_grf_csv data/sample/grf_trial01.csv --pred_grf_csv out/predicted_grf.csv --bw_kg 70
```

## Folder Structure
- `src/`: Source code for data processing, feature extraction, modeling, and prediction.
- `tests/`: Unit tests for the codebase.
- `data/`: Directory for storing input and output data files.

## Limitations
- The current implementation uses synthetic data and simplistic models.
- Future integration with OpenSim for more advanced biomechanical analysis is planned.

## Next Steps
- Integrate with OpenSim for IK/ID analysis.
- Enhance models with more complex algorithms and real-world data.

---

# Tremor vs Steady Classifier — From Dev to Sahaana and Krisha

Hi Sahaana and Krisha — it’s Dev! This section is just for you. We’re going to learn how to teach a computer to look at tiny wrist movements and decide if the hand is “steady” or if it has a “tremor.” We’ll use a special notebook file called `tremor_classifier_nb.ipynb` that lives in this folder.

Folder location on this computer:

- `c:\Users\locha\Downloads\TremorClassifier`

## What are we trying to do?
- **Goal**: Build a simple AI that looks at short clips (about 2 seconds) of wrist motion and says “steady” or “tremor.”
- **Why this matters**: Detecting tremor can help doctors track symptoms and help people over time.

## Where does the motion data come from?
- **Smartwatch sensors**: A watch measures motion in two ways:
  - Accelerometer (ax, ay, az): how fast your hand is speeding up in 3 directions.
  - Gyroscope (gx, gy, gz): how fast your hand is turning in 3 directions.
- We use these 6 signals together. Think of them like 6 “eyes” watching your hand move.

Images to help you picture this:

![Sensor Axes](images/axes_explainer.png)

And here’s what “steady” vs “tremor” can look like as a simple signal (like a wavy line over time):

![Steady vs Tremor](images/steady_vs_tremor.gif)

If the GIF doesn’t show, look for the static image instead: `images/steady_vs_tremor.png`.

## Two ways to get data
1. **Simulated data** (computer-made): Good for learning and quick tests.
2. **Real data (PADS dataset)**: From the Parkinson’s Disease Smartwatch dataset on PhysioNet.

## How the notebook works (step by step)
1. Load data (simulated, CSV you provide, or PADS dataset files).
2. Draw a picture of overall movement (a quick plot of acceleration size over time).
3. Slice the data into small windows (2 seconds long), like breaking a song into tiny parts.
4. For each window, compute simple “features” (tiny summaries):
   - Mean: the average value.
   - Standard deviation: how wiggly it is.
   - RMS: like average power/energy.
   - Dominant frequency: the strongest “beat” in the window (found using FFT).
5. Train two simple models to tell steady vs tremor:
   - Logistic Regression: draws one straight line between the two groups.
   - Decision Tree: learns if/then rules (like “if frequency > X, say tremor”).
6. Check which model is more accurate and show a confusion matrix (a tiny scoreboard).
7. Save the best model so you can use it later to classify new data.

## What is a “feature” and why do we need it?
- A feature is a small number that tells something important about the data.
- Example: If you clap your hands evenly, the “dominant frequency” is like the beat you clap at. Tremor often has a beat around 3–8 times per second (Hz). The model uses these clues to decide.

## What is a “model” and how does it learn?
- A model is a recipe the computer uses to make decisions.
- **Logistic Regression**: Tries to draw a straight line that separates “steady” from “tremor.”
- **Decision Tree**: Asks simple questions like “Is the frequency above 4.5 Hz?” If yes, go right; if no, go left.
- The notebook trains both and picks the winner by accuracy.

## Open and run the notebook
File: `tremor_classifier_nb.ipynb`

### Easiest way (one-click-ish): use the run script
1. Open Windows PowerShell in this folder: `c:\Users\locha\Documents\safestride`
2. Run:
   ```powershell
   .\run_me.ps1
   ```
   - This creates a Python environment, installs what we need, generates the pictures and GIF above, and opens the notebook for you.
   - If PowerShell blocks the script, right-click the file and choose “Run with PowerShell,” or run PowerShell “As Administrator.”

### Option A: Simulated data (easiest)
1. Open the notebook.
2. In the first cell, make sure:
   ```python
   DATA_SOURCE = 'simulate'
   ```
3. Run all the cells from top to bottom.
4. You’ll see accuracies, confusion matrices, and a saved model in `models/best_tremor_model.joblib`.

### Option B: Your own CSV
Your CSV must have columns: `time_ms, ax, ay, az, gx, gy, gz, label` (label is "steady" or "tremor").
1. Save your CSV as `hand_motion.csv` in this folder.
2. Set:
   ```python
   DATA_SOURCE = 'csv'
   ```
3. Run all cells.

### Option C: PhysioNet PADS dataset (real smartwatch data)
1. Download the ZIP: https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/
2. Unzip to: `pads_dataset` inside this folder (you should see `pads_dataset/movement/timeseries/*.txt`).
3. In the notebook, set:
   ```python
   DATA_SOURCE = 'pads'
   pads_root = Path('pads_dataset')
   pads_max_files = 100  # optional: use fewer files to run faster
   ```
4. Run all cells.
5. Because PADS doesn’t have per-window labels, the notebook uses a simple rule (a “heuristic”):
   - If the gyro’s dominant frequency is between 3.5 and 8 Hz and its energy is big enough, call it “tremor”; else “steady.”
6. You can adjust the rule to get a better balance of labels:
   ```python
   tremor_freq_range = (3.5, 8.0)  # widen to (3.0, 9.0) if needed
   tremor_gyro_rms_min = 1.0       # lower to 0.8 or 0.6 if everything becomes "steady"
   ```

After training, the notebook also saves an image of the confusion matrices here:

- `images/confusion_matrices.png`

## How does prediction work?
- After training, the best model is saved.
- To classify a new 2-second window, we compute the same features and ask the model for a decision. The notebook shows a small demo at the end.

## Try changing things!
- Change `window_seconds` to 1.0 or 3.0 and see what happens.
- Try `pads_max_files = 20` vs `200`. More data can help but takes longer.
- Compare the two models. Does the Decision Tree do better or worse than Logistic Regression?

## Troubleshooting
- “Only one class found” warning: Loosen the tremor thresholds so you have both “steady” and “tremor.”
- Slow or memory heavy: Lower `pads_max_files`.
- No files found: Check the folder path `pads_dataset/movement/timeseries` exists.

## Little glossary
- **Accelerometer**: Measures how fast something speeds up.
- **Gyroscope**: Measures how fast something turns.
- **RMS**: A way to measure average power/energy of a signal.
- **Frequency (Hz)**: How many times something repeats each second.
- **FFT**: A math tool to find the strongest frequencies (the main “beats”) in a signal.
- **Accuracy**: How often the model’s guesses are correct.
- **Confusion matrix**: A small table that shows correct and incorrect guesses.

## Why did we choose these features and models?
- Tremor often has a clear “beat” (frequency), so frequency features help a lot.
- RMS and std tell us how strong or shaky the motion is.
- We start with simple models because they’re easy to understand and usually work well on small problems. Then, if we need more power, we can try advanced models later.

## Be kind and careful (Ethics)
- These tools can help people, but they’re not perfect.
- Real medical decisions should be made by doctors. We use this as a learning tool and a first step.

I hope you have fun exploring how AI can learn from motion! 😊 — Dev
=======
# SafeStride
>>>>>>> f0cb1e8f0c850c60d180d7901221de57c83c71f9
