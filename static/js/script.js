// Heart Disease Prediction App - JavaScript

document.addEventListener("DOMContentLoaded", function () {
  // Initialize the application
  initializeApp();
});

function initializeApp() {
  // Setup form submission
  setupFormSubmission();

  // Setup smooth scrolling
  setupSmoothScrolling();

  // Setup form validation
  setupFormValidation();

  // Setup tooltips and popovers
  setupBootstrapComponents();

  console.log("Heart Disease Prediction App initialized successfully!");
}

function setupFormSubmission() {
  const form = document.getElementById("predictionForm");
  const submitBtn = document.getElementById("predictBtn");

  if (form && submitBtn) {
    form.addEventListener("submit", handleFormSubmission);

    // Add loading state management
    submitBtn.addEventListener("click", function (e) {
      if (!form.checkValidity()) {
        e.preventDefault();
        e.stopPropagation();
        form.classList.add("was-validated");

        // Scroll to first invalid field
        const firstInvalid = form.querySelector(":invalid");
        if (firstInvalid) {
          firstInvalid.scrollIntoView({
            behavior: "smooth",
            block: "center",
          });
          firstInvalid.focus();
        }

        showAlert("Please fill in all required fields correctly.", "warning");
      }
    });
  }
}

async function handleFormSubmission(event) {
  event.preventDefault();

  const form = event.target;
  const submitBtn = document.getElementById("predictBtn");
  const loadingOverlay = document.getElementById("loadingOverlay");

  // Validate form
  if (!form.checkValidity()) {
    form.classList.add("was-validated");
    return;
  }

  try {
    // Show loading state
    setLoadingState(true, submitBtn, loadingOverlay);

    // Collect form data
    const formData = new FormData(form);

    // Validate numerical inputs
    if (!validateNumericalInputs(formData)) {
      throw new Error("Please check your numerical inputs for valid ranges.");
    }

    // Make prediction request
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(
        errorData.error || "Prediction failed. Please try again."
      );
    }

    const result = await response.json();

    // Display results
    displayResults(result);

    // Scroll to results
    document.getElementById("results").scrollIntoView({
      behavior: "smooth",
    });

    // Show success message
    showAlert("Prediction completed successfully!", "success");
  } catch (error) {
    console.error("Prediction error:", error);
    showAlert(
      error.message || "An error occurred during prediction. Please try again.",
      "danger"
    );
  } finally {
    // Hide loading state
    setLoadingState(false, submitBtn, loadingOverlay);
  }
}

function validateNumericalInputs(formData) {
  const validationRules = {
    age: { min: 1, max: 120 },
    trestbps: { min: 80, max: 250 },
    chol: { min: 100, max: 600 },
    thalch: { min: 60, max: 220 },
    oldpeak: { min: 0, max: 10 },
    ca: { min: 0, max: 3 },
  };

  for (const [field, rules] of Object.entries(validationRules)) {
    const value = parseFloat(formData.get(field));

    if (isNaN(value) || value < rules.min || value > rules.max) {
      console.error(`Invalid value for ${field}: ${value}`);
      return false;
    }
  }

  return true;
}

function setLoadingState(isLoading, submitBtn, loadingOverlay) {
  if (isLoading) {
    submitBtn.disabled = true;
    submitBtn.innerHTML =
      '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
    if (loadingOverlay) {
      loadingOverlay.style.display = "flex";
    }
  } else {
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<i class="fas fa-brain me-2"></i>Analyze with AI';
    if (loadingOverlay) {
      loadingOverlay.style.display = "none";
    }
  }
}

function displayResults(result) {
  const resultsSection = document.getElementById("results");
  const resultsContent = document.getElementById("resultsContent");

  if (!resultsSection || !resultsContent) return;

  // Clear previous results
  resultsContent.innerHTML = "";

  // Create results HTML
  const resultsHTML = createResultsHTML(result);
  resultsContent.innerHTML = resultsHTML;

  // Show results section
  resultsSection.style.display = "block";

  // Add animation classes
  setTimeout(() => {
    const resultItems = resultsContent.querySelectorAll(".result-item");
    resultItems.forEach((item, index) => {
      setTimeout(() => {
        item.style.opacity = "1";
        item.style.transform = "translateY(0)";
      }, index * 200);
    });
  }, 100);
}

function createResultsHTML(result) {
  const svmPositive = result.svm_prediction > 0;
  const knnPositive = result.knn_prediction > 0;
  const agreement = result.agreement === "Yes";

  return `
        <div class="col-lg-6 mb-4">
            <div class="result-item ${
              svmPositive ? "result-positive" : "result-negative"
            }">
                <div class="text-center">
                    <div class="result-icon">
                        <i class="fas fa-vector-square ${
                          svmPositive ? "text-danger" : "text-success"
                        }"></i>
                    </div>
                    <h4>Support Vector Machine (SVM)</h4>
                    <div class="result-badge ${
                      svmPositive ? "bg-danger" : "bg-success"
                    } text-white p-2 rounded mb-3">
                        ${result.svm_result}
                    </div>
                    <p class="text-muted">
                        Prediction: <strong>${
                          svmPositive
                            ? "Heart Disease Detected"
                            : "No Heart Disease"
                        }</strong>
                    </p>
                    ${
                      result.svm_confidence
                        ? `
                        <div class="confidence-info">
                            <small class="text-muted">Confidence: ${(
                              result.svm_confidence * 100
                            ).toFixed(1)}%</small>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${
                                  result.svm_confidence * 100
                                }%"></div>
                            </div>
                        </div>
                    `
                        : ""
                    }
                </div>
            </div>
        </div>
        
        <div class="col-lg-6 mb-4">
            <div class="result-item ${
              knnPositive ? "result-positive" : "result-negative"
            }">
                <div class="text-center">
                    <div class="result-icon">
                        <i class="fas fa-users ${
                          knnPositive ? "text-danger" : "text-success"
                        }"></i>
                    </div>
                    <h4>K-Nearest Neighbors (KNN)</h4>
                    <div class="result-badge ${
                      knnPositive ? "bg-danger" : "bg-success"
                    } text-white p-2 rounded mb-3">
                        ${result.knn_result}
                    </div>
                    <p class="text-muted">
                        Prediction: <strong>${
                          knnPositive
                            ? "Heart Disease Detected"
                            : "No Heart Disease"
                        }</strong>
                    </p>
                    ${
                      result.knn_confidence
                        ? `
                        <div class="confidence-info">
                            <small class="text-muted">Confidence: ${(
                              result.knn_confidence * 100
                            ).toFixed(1)}%</small>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${
                                  result.knn_confidence * 100
                                }%"></div>
                            </div>
                        </div>
                    `
                        : ""
                    }
                </div>
            </div>
        </div>
        
        <div class="col-12">
            <div class="result-item">
                <div class="text-center">
                    <div class="result-icon">
                        <i class="fas fa-handshake ${
                          agreement ? "text-success" : "text-warning"
                        }"></i>
                    </div>
                    <h4>Algorithm Agreement</h4>
                    <div class="result-badge ${
                      agreement ? "bg-success" : "bg-warning"
                    } text-white p-2 rounded mb-3">
                        ${
                          agreement
                            ? "Both algorithms agree"
                            : "Algorithms disagree"
                        }
                    </div>
                    <p class="text-muted">
                        ${
                          agreement
                            ? "Both SVM and KNN models produced the same prediction, increasing confidence in the result."
                            : "The algorithms produced different predictions. Consider consulting a healthcare professional for further evaluation."
                        }
                    </p>
                    ${
                      !agreement
                        ? `
                        <div class="alert alert-warning mt-3">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            <strong>Important:</strong> When algorithms disagree, it may indicate a borderline case. 
                            Please consult with a healthcare professional for proper medical advice.
                        </div>
                    `
                        : ""
                    }
                </div>
            </div>
        </div>
        
        <div class="col-12 mt-4">
            <div class="alert alert-info">
                <h5><i class="fas fa-info-circle me-2"></i>Important Notes:</h5>
                <ul class="mb-0">
                    <li>This prediction is for educational purposes only</li>
                    <li>Always consult healthcare professionals for medical decisions</li>
                    <li>Regular health check-ups are recommended regardless of the prediction</li>
                    <li>The algorithms are trained on historical data and may not account for all factors</li>
                </ul>
            </div>
        </div>
    `;
}

function resetForm() {
  const form = document.getElementById("predictionForm");
  const resultsSection = document.getElementById("results");

  if (form) {
    form.reset();
    form.classList.remove("was-validated");
  }

  if (resultsSection) {
    resultsSection.style.display = "none";
  }

  // Scroll back to form
  document.getElementById("prediction-form").scrollIntoView({
    behavior: "smooth",
  });

  showAlert("Form reset successfully. You can now enter new data.", "info");
}

function setupSmoothScrolling() {
  // Handle anchor links
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute("href"));
      if (target) {
        target.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }
    });
  });
}

function setupFormValidation() {
  // Add real-time validation feedback
  const inputs = document.querySelectorAll(".form-control, .form-select");

  inputs.forEach((input) => {
    input.addEventListener("blur", function () {
      validateInput(this);
    });

    input.addEventListener("input", function () {
      if (this.classList.contains("is-invalid")) {
        validateInput(this);
      }
    });
  });
}

function validateInput(input) {
  const isValid = input.checkValidity();

  if (isValid) {
    input.classList.remove("is-invalid");
    input.classList.add("is-valid");
  } else {
    input.classList.remove("is-valid");
    input.classList.add("is-invalid");
  }

  return isValid;
}

function setupBootstrapComponents() {
  // Initialize tooltips
  const tooltipTriggerList = [].slice.call(
    document.querySelectorAll('[data-bs-toggle="tooltip"]')
  );
  tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
  });

  // Initialize popovers
  const popoverTriggerList = [].slice.call(
    document.querySelectorAll('[data-bs-toggle="popover"]')
  );
  popoverTriggerList.map(function (popoverTriggerEl) {
    return new bootstrap.Popover(popoverTriggerEl);
  });
}

function showAlert(message, type = "info") {
  // Remove existing alerts
  const existingAlerts = document.querySelectorAll(".dynamic-alert");
  existingAlerts.forEach((alert) => alert.remove());

  // Create alert element
  const alertDiv = document.createElement("div");
  alertDiv.className = `alert alert-${type} alert-dismissible fade show dynamic-alert`;
  alertDiv.style.position = "fixed";
  alertDiv.style.top = "100px";
  alertDiv.style.right = "20px";
  alertDiv.style.zIndex = "9999";
  alertDiv.style.minWidth = "300px";
  alertDiv.style.maxWidth = "500px";

  alertDiv.innerHTML = `
        <i class="fas fa-${getAlertIcon(type)} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

  // Add to page
  document.body.appendChild(alertDiv);

  // Auto-dismiss after 5 seconds
  setTimeout(() => {
    if (alertDiv.parentNode) {
      alertDiv.remove();
    }
  }, 5000);
}

function getAlertIcon(type) {
  const icons = {
    success: "check-circle",
    danger: "exclamation-triangle",
    warning: "exclamation-circle",
    info: "info-circle",
  };
  return icons[type] || "info-circle";
}

// Utility functions
function formatNumber(num, decimals = 1) {
  return Number(num).toFixed(decimals);
}

function getHealthRiskLevel(confidence) {
  if (confidence < 0.3) return "Low";
  if (confidence < 0.7) return "Moderate";
  return "High";
}

// Export functions for global access
window.resetForm = resetForm;
window.showAlert = showAlert;

// Health check on page load
window.addEventListener("load", function () {
  // Check if the backend is responsive
  fetch("/api/health")
    .then((response) => response.json())
    .then((data) => {
      if (data.status === "healthy") {
        console.log("✅ Backend health check passed");
        if (!data.models_loaded) {
          showAlert(
            "⚠️ Models not loaded. Some features may not work properly.",
            "warning"
          );
        }
      }
    })
    .catch((error) => {
      console.error("❌ Backend health check failed:", error);
      showAlert(
        "⚠️ Backend connection issue. Please refresh the page.",
        "warning"
      );
    });
});

// Progressive Web App features
if ("serviceWorker" in navigator) {
  window.addEventListener("load", function () {
    // You can add service worker registration here for offline functionality
    console.log("Service Worker support detected");
  });
}

// Keyboard shortcuts
document.addEventListener("keydown", function (e) {
  // Ctrl/Cmd + Enter to submit form
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    const form = document.getElementById("predictionForm");
    if (
      form &&
      document.activeElement &&
      form.contains(document.activeElement)
    ) {
      form.requestSubmit();
    }
  }

  // Escape to close alerts
  if (e.key === "Escape") {
    const alerts = document.querySelectorAll(".dynamic-alert");
    alerts.forEach((alert) => alert.remove());
  }
});
