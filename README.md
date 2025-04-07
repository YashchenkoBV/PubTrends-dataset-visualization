# PubTrends-dataset-visualization

## Description
This is a Flask-based web application that does clustering of GEO datasets, associated with a given list of PMIDs.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YashchenkoBV/PubTrends-dataset-visualization.git
   cd PubTrends-dataset-visualization
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Submit a list of PMIDs (.txt)

5. Wait for information retrieval

6. See the visualization

## Requirements
Flask
flask-socketio
numpy
pandas
scikit-learn
scipy
plotly
matplotlib
requests
beautifulsoup4
lxml

All the required packages (with the appropriate versions) are specified in the `requirements.txt` file.

## Features
- **Interactive visualization** - see the plot for particular clusters or click on dots to view info about the corresponding GEO dataset
- **Several projects** - submit different lists of PMIDs and choose file names to store the data about the datasets and visualizations in
- **Dynamic caching** - retrieved data is saved dynamically
- **Progress bar** - see the status of current job on the progress bar or in the logs window
