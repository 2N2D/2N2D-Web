"use server";

export async function startOptimization(
  selectedInputs: String[],
  targetFeature: String,
  epochs: number,
) {
  if (!targetFeature) {
    return;
  }

  const progressBar = document.getElementById("opt-progress-bar");
  const progressText = document.getElementById("opt-progress-text");

  if (progressBar) progressBar.style.width = "0%";
  if (progressText) progressText.textContent = "Starting optimization...";

  console.log("Calling Python function directly...");

  try {
    return await fetch("http://127.0.0.1:8000/optimize ", {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify({
        input_features: selectedInputs,
        target_feature: targetFeature,
        max_epochs: epochs,
      }),
    });
  } catch (error) {
    console.error("Error during optimization:", error);
    throw error;
  }
}

export async function sendCSV(formData: any) {
  try {
    const result = await fetch("http://127.0.0.1:8000/upload-csv", {
      method: "POST",
      body: formData,
    });

    const response = await result.json();
    console.log(response);
    return response;
  } catch (error) {
    console.error("Upload failed", error);
  }
}
