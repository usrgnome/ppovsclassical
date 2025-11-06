import React from "react";

function LoadingScreen({ message = "Loading..." }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50 font-sans p-6">
      <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-blue-600 mb-4"></div>
      <p className="text-gray-700 text-lg">{message}</p>
    </div>
  );
}

export default LoadingScreen;