import React, { useState } from "react";
import "@fortawesome/fontawesome-free/css/all.min.css";


function App() {
  const [totalHouseholds, setTotalHouseholds] = useState('');
  const [coveredHouseholds, setCoveredHouseholds] = useState('');
  const [zoneName, setZoneName] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handlePredict = async () => {
    setError('');
    setResult(null);

    const data = {
      total_households: totalHouseholds,
      covered_households: coveredHouseholds,
      zone_name: zoneName
    };

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });

      const json = await response.json();
      if (response.ok) {
        setResult(json);
      } else {
        setError(json.error || 'Prediction failed.');
      }
    } catch (err) {
      setError('Error connecting to backend.');
    }
  };

  return (
    <div className="min-h-screen bg-blueGray-100 flex items-center justify-center px-4">
      <div className="w-full max-w-2xl bg-white rounded-2xl shadow-xl p-8">
        <h2 className="text-3xl font-semibold mb-6 text-center text-emerald-600">
          Waste Source Segregation Predictor
        </h2>

        <div className="grid grid-cols-1 gap-6">
          <div>
            <label className="block text-blueGray-600 text-sm font-bold mb-2">
              Total Households
            </label>
            <input
              type="number"
              value={totalHouseholds}
              onChange={(e) => setTotalHouseholds(e.target.value)}
              placeholder="e.g., 1000"
              className="px-4 py-2 border border-blueGray-300 rounded-lg w-full focus:outline-none focus:ring-2 focus:ring-emerald-500"
            />
          </div>

          <div>
            <label className="block text-blueGray-600 text-sm font-bold mb-2">
              Covered Households
            </label>
            <input
              type="number"
              value={coveredHouseholds}
              onChange={(e) => setCoveredHouseholds(e.target.value)}
              placeholder="e.g., 800"
              className="px-4 py-2 border border-blueGray-300 rounded-lg w-full focus:outline-none focus:ring-2 focus:ring-emerald-500"
            />
          </div>

          <div>
            <label className="block text-blueGray-600 text-sm font-bold mb-2">
              Zone Name
            </label>
            <input
              type="text"
              value={zoneName}
              onChange={(e) => setZoneName(e.target.value)}
              placeholder="e.g., Zone 1"
              className="px-4 py-2 border border-blueGray-300 rounded-lg w-full focus:outline-none focus:ring-2 focus:ring-emerald-500"
            />
          </div>

          <button
            onClick={handlePredict}
            className="bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-2 px-4 rounded-lg transition duration-300"
          >
            Predict
          </button>

          {result && (
            <div className="mt-6 p-4 bg-emerald-100 border border-emerald-400 text-emerald-700 rounded-lg shadow">
              <h4 className="font-semibold">Predicted HH Source Segregation:</h4>
              <p className="text-2xl font-bold mt-1">
                {result.HH_Source_Segregation.toFixed(2)}
              </p>
            </div>
          )}

          {error && (
            <div className="mt-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg shadow">
              <p>{error}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
