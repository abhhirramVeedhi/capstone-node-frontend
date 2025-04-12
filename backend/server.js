// const express = require("express");
// const app = express();
// const multer = require("multer");
// const mongoose = require("mongoose");
// const bodyParser = require("body-parser");
// const path = require("path");
// const fs = require("fs");
// const fetch = (...args) =>
//   import("node-fetch").then(({ default: fetch }) => fetch(...args));
// const FormData = require("form-data");

// app.set("view engine", "ejs");
// app.set("views", path.join(__dirname, "../frontend/views"));

// app.use(express.static(path.join(__dirname, "../public")));
// app.use("/uploads", express.static(path.join(__dirname, "../uploads")));
// app.use("/heatmaps", express.static(path.join(__dirname, "../heatmaps")));

// app.use(bodyParser.urlencoded({ extended: true }));

// mongoose.connect("mongodb://127.0.0.1:27017/deepfakeDB");

// const detectionSchema = new mongoose.Schema({
//   filename: String,
//   prediction: String,
//   confidence: Number,
//   date: { type: Date, default: Date.now },
//   feedback: String,
// });

// const Detection = mongoose.model("Detection", detectionSchema);

// const storage = multer.diskStorage({
//   destination: (req, file, cb) => cb(null, path.join(__dirname, "../uploads")),
//   filename: (req, file, cb) => cb(null, Date.now() + "-" + file.originalname),
// });
// const upload = multer({ storage: storage });

// // Home route
// app.get("/", (req, res) => {
//   res.render("home", { result: null });
// });

// // Detect (GET page)
// app.get("/detect", (req, res) => {
//   res.render("detect", { result: null });
// });

// // Detect (POST logic)
// app.post("/detect", upload.single("file"), async (req, res) => {
//   try {
//     if (!req.file) throw new Error("No file uploaded");

//     const { filename, path: filepath } = req.file;
//     const formData = new FormData();
//     formData.append("image", fs.createReadStream(filepath));

//     const response = await fetch("http://127.0.0.1:5000/predict", {
//       method: "POST",
//       body: formData,
//       headers: formData.getHeaders(),
//     });

//     if (!response.ok)
//       throw new Error(`Flask API Error: ${response.statusText}`);

//     const data = await response.json();

//     const newEntry = new Detection({
//       filename,
//       prediction: data.label,
//       confidence: data.confidence,
//     });
//     await newEntry.save();

//     res.render("detect", {
//       result: {
//         prediction: data.label,
//         confidence: data.confidence,
//         filename,
//         heatmap: data.heatmap,
//       },
//     });
//   } catch (error) {
//     console.error("❌ Upload error:", error.message);
//     res.status(500).send("Upload failed: " + error.message);
//   }
// });

// // Dashboard
// app.get("/dashboard", async (req, res) => {
//   const logs = await Detection.find().sort({ date: -1 });
//   const real = logs.filter((e) => e.prediction === "Real Image").length;
//   const fake = logs.filter((e) => e.prediction === "Fake Image").length;
//   res.render("dashboard", { logs, real, fake });
// });

// // Feedback route
// app.post("/feedback", async (req, res) => {
//   const { filename, feedback } = req.body;
//   await Detection.findOneAndUpdate({ filename }, { feedback });
//   res.redirect("/dashboard");
// });

// // Learn
// app.get("/learn", (req, res) => {
//   res.render("learn");
// });

// app.listen(3002, () => {
//   console.log("✅ Server running on http://localhost:3002");
// });

const express = require("express");
const app = express();
const multer = require("multer");
const mongoose = require("mongoose");
const bodyParser = require("body-parser");
const path = require("path");
const fs = require("fs");
const fetch = (...args) =>
  import("node-fetch").then(({ default: fetch }) => fetch(...args));
const FormData = require("form-data");

app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "../views"));

express.static(path.join(__dirname, "../public"))

app.use("/uploads", express.static(path.join(__dirname, "../uploads")));
app.use("/heatmaps", express.static(path.join(__dirname, "../heatmaps")));

app.use(bodyParser.urlencoded({ extended: true }));

mongoose.connect("mongodb://127.0.0.1:27017/deepfakeDB");

const detectionSchema = new mongoose.Schema({
  filename: String,
  prediction: String,
  confidence: Number,
  date: { type: Date, default: Date.now },
  feedback: String,
});

const Detection = mongoose.model("Detection", detectionSchema);

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, path.join(__dirname, "../uploads")),
  filename: (req, file, cb) => cb(null, Date.now() + "-" + file.originalname),
});
const upload = multer({ storage: storage });

// Home route
app.get("/", (req, res) => {
  res.render("home", { result: null });
});

// Detect (GET page)
app.get("/detect", (req, res) => {
  res.render("detect", { result: null });
});

// Detect (POST logic)
app.post("/detect", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) throw new Error("No file uploaded");

    const { filename, path: filepath } = req.file;
    const formData = new FormData();
    formData.append("image", fs.createReadStream(filepath));

    const response = await fetch("https://capstone-node-frontend.onrender.com/predict", {
      method: "POST",
      body: formData,
      headers: formData.getHeaders(),
    });

    if (!response.ok)
      throw new Error(`Flask API Error: ${response.statusText}`);

    const data = await response.json();

    const newEntry = new Detection({
      filename,
      prediction: data.label,
      confidence: data.confidence,
    });
    await newEntry.save();

    res.render("detect", {
      result: {
        prediction: data.label,
        confidence: data.confidence,
        filename,
        heatmap: data.heatmap,
      },
    });
  } catch (error) {
    console.error("❌ Upload error:", error.message);
    res.status(500).send("Upload failed: " + error.message);
  }
});

// Dashboard
app.get("/dashboard", async (req, res) => {
  try {
    const logs = await Detection.find().sort({ date: -1 });
    const real = logs.filter((e) => e.prediction === "Real Image").length;
    const fake = logs.filter((e) => e.prediction === "Fake Image").length;

    // Calculate confidence distribution
    let highConfidence = 0;
    let mediumConfidence = 0;
    let lowConfidence = 0;

    if (logs.length > 0) {
      // Count images in each confidence category
      const highCount = logs.filter((log) => log.confidence >= 80).length;
      const mediumCount = logs.filter(
        (log) => log.confidence >= 50 && log.confidence < 80
      ).length;
      const lowCount = logs.filter((log) => log.confidence < 50).length;

      // Calculate percentages (rounded to nearest integer)
      highConfidence = Math.round((highCount / logs.length) * 100) || 0;
      mediumConfidence = Math.round((mediumCount / logs.length) * 100) || 0;

      // Make sure percentages add up to 100%
      lowConfidence = 100 - highConfidence - mediumConfidence;

      // Handle edge case to ensure lowConfidence is not negative
      if (lowConfidence < 0) {
        lowConfidence = 0;
        // Adjust the larger value to compensate
        if (highConfidence > mediumConfidence) {
          highConfidence = 100 - mediumConfidence;
        } else {
          mediumConfidence = 100 - highConfidence;
        }
      }
    }

    // Debug output to verify values are calculated correctly
    console.log("Passing to template:", {
      logsCount: logs.length,
      real,
      fake,
      highConfidence,
      mediumConfidence,
      lowConfidence,
    });

    res.render("dashboard", {
      logs,
      real,
      fake,
      highConfidence,
      mediumConfidence,
      lowConfidence,
    });
  } catch (error) {
    console.error("❌ Dashboard error:", error.message);
    res.status(500).send("Dashboard error: " + error.message);
  }
});

// Feedback route
app.post("/feedback", async (req, res) => {
  const { filename, feedback } = req.body;
  await Detection.findOneAndUpdate({ filename }, { feedback });
  res.redirect("/dashboard");
});

// Learn
app.get("/learn", (req, res) => {
  res.render("learn");
});

app.listen(3002, () => {
  console.log("✅ Server running on http://localhost:3002");
});

// const serverless = require("serverless-http");
// module.exports.handler = serverless(app);

