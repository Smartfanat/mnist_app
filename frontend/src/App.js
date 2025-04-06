// frontend/src/App.js
import React, { useState, useRef } from 'react';
import DigitCanvas from './DigitCanvas';
import './App.css';

const BACKEND_URL = 'http://localhost:5001';

function App() {
    const [prediction, setPrediction] = useState(null);
    const [probabilities, setProbabilities] = useState(null);
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const canvasRef = useRef(null);

    const handlePredict = async () => {
        if (!canvasRef.current) {
            setError('Canvas is not ready.');
            return;
        }

        const imageDataUrl = canvasRef.current.getImageDataUrl();
        if (!imageDataUrl) {
            setError('Could not get image data from canvas.');
            return;
        }

        setPrediction(null);
        setProbabilities(null);
        setError('');
        setIsLoading(true);

        console.log('Sending image data to backend...');

        try {
            const response = await fetch(`${BACKEND_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ imageDataUrl: imageDataUrl }),
            });

            if (!response.ok) {
                let errorMsg = `Error: ${response.status} ${response.statusText}`;
                try {
                    const errorData = await response.json();
                    errorMsg = errorData.error || errorMsg;
                } catch (e) {
                    console.error("Could not parse error response JSON:", e);
                }
                throw new Error(errorMsg);
            }

            const data = await response.json();

            console.log('Received prediction data:', data);

            setPrediction(data.prediction); //
            setProbabilities(data.probabilities);

        } catch (err) {
            console.error("Prediction request failed:", err);
            setError(err.message || 'Prediction request failed. Is the backend running and accessible?');
        } finally {
            setIsLoading(false);
        }
    };

    const handleClear = () => {
        if (canvasRef.current) {
            canvasRef.current.clearCanvas();
        }

        setPrediction(null);
        setProbabilities(null);
        setError('');
        setIsLoading(false);
    };

    return (
        <div className="App">
            <h1>MNIST Digit Predictor</h1>
            <p>Draw a single digit (0-9) in the box below:</p>

            {/* Container for the canvas */}
            <div className="canvas-container">
                {/* The drawing canvas component, passing the ref */}
                <DigitCanvas ref={canvasRef} />
            </div>

            {/* Container for the control buttons */}
            <div className="controls">
                <button onClick={handlePredict} disabled={isLoading}>
                    {/* Show different text on the button while loading */}
                    {isLoading ? 'Predicting...' : 'Predict Digit'}
                </button>
                <button onClick={handleClear} disabled={isLoading}>
                    Clear Canvas
                </button>
            </div>

            {/* Display error messages if any */}
            {error && <p className="error-message">{error}</p>}

            {/* Display prediction results only if a prediction has been made */}
            {prediction !== null && (
                <div className="results">
                    {/* Display the top predicted digit */}
                    <h2>Prediction: <span className="prediction-digit">{prediction}</span></h2>

                    {/* Display probabilities if available */}
                    {probabilities && (
                        <div className="probabilities">
                            <h3>Top 3 Predictions:</h3>
                            <ul>
                                {Object.entries(probabilities) // ["0": 0.01, "1": 0.03, ...] -> [ ["0", 0.01], ["1", 0.03], ...]
                                      .sort(([, probA], [, probB]) => probB - probA) // Sort by probability (value) descending
                                      .slice(0, 3) // Take the top 3 entries from the sorted array
                                      .map(([digit, prob]) => ( // Map over only the top 3
                                          <li key={digit}>
                                              Digit {digit}: <strong>{(prob * 100).toFixed(2)}%</strong>
                                          </li>
                                      ))
                                }
                            </ul>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

export default App;