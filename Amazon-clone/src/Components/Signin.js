import { React, useState } from "react";
import "./signin.css";
import Logo from "../imgs/logo2.png";
import BG1 from "../imgs/login-BG.png";
import BG2 from "../imgs/login-BG2.png";
import google from "../imgs/google.png";
import { Link, useNavigate } from "react-router-dom";
import { app } from "../Firebase";
import {
  getAuth,
  signInWithEmailAndPassword,
  GoogleAuthProvider,
  signInWithPopup,
} from "firebase/auth";
import swal from "sweetalert";

const auth = getAuth(app);
const provider = new GoogleAuthProvider();

function Signin() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [emailError, setEmailError] = useState("");
  const [PasswordError, setPasswordError] = useState("");
  const [bgLoaded, setBgLoaded] = useState(false);
  const [role, setRole] = useState("user"); // "user" or "seller"
  const navigate = useNavigate();

  document.title = "Amazon";

  const handleEmailChange = (event) => setEmail(event.target.value);
  const handlePasswordChange = (event) => setPassword(event.target.value);
  const handleBgLoad = () => setBgLoaded(true);

  const handleEmailBlur = () => {
    if (!email || !email.includes("@") || !email.includes(".com")) {
      setEmailError("Please enter a valid email address.");
    } else {
      setEmailError("");
    }
  };

  const handlePasswordBlur = () => {
    if (!password) {
      setPasswordError("Please enter your password.");
    } else if (password.length < 4) {
      setPasswordError("Password is too short.");
    } else {
      setPasswordError("");
    }
  };

  const LogInUser = async () => {
    signInWithEmailAndPassword(auth, email, password)
      .then(() => {
        if (role === "user") {
          navigate("/products");
        } else {
          navigate("/seller-dashboard"); // you can create this page separately
        }
      })
      .catch((error) => {
        swal({
          title: "Error!",
          text: error.message,
          icon: "error",
          buttons: "Ok",
        });
      });
  };

  const GoogleAuth = async () => {
    signInWithPopup(auth, provider)
      .then(() => {
        if (role === "user") {
          navigate("/products");
        } else {
          navigate("/seller-dashboard");
        }
      })
      .catch((error) => {
        swal({
          title: "Error!",
          text: error.message,
          icon: "error",
          buttons: "Ok",
        });
      });
  };

  return (
    <div className="signin-page">
      <div className="login-navbar">
        <div className="main-logo">
          <img src={Logo} className="amazon-logo" alt="logo" />
        </div>
        <div className="signup">
          <Link to="/signup">
            <button className="signup-btn">Sign up</button>
          </Link>
        </div>
      </div>

      <div className="background">
        <img src={BG1} className="BG1" onLoad={handleBgLoad} alt="bg1" />
        <img src={BG2} className="BG2" onLoad={handleBgLoad} alt="bg2" />
      </div>

      {bgLoaded && (
        <div className="main-form">
          <div className="login-form">
            <div className="some-text">
              <p className="user">Sign in as {role === "user" ? "User" : "Seller"}</p>
              <p className="user-desc">
                Enter your credentials to continue
              </p>
              <div className="role-switch">
                <button
                  className={role === "user" ? "role-active" : ""}
                  onClick={() => setRole("user")}
                >
                  User
                </button>
                <button
                  className={role === "seller" ? "role-active" : ""}
                  onClick={() => setRole("seller")}
                >
                  Seller
                </button>
              </div>
            </div>

            <div className="user-details">
              <input
                type="email"
                placeholder="Enter Email"
                className="email"
                value={email}
                onChange={handleEmailChange}
                onBlur={handleEmailBlur}
                required
              />
              {emailError && <div className="error-message">{emailError}</div>}

              <input
                type="password"
                placeholder="Password"
                className="password"
                value={password}
                onChange={handlePasswordChange}
                onBlur={handlePasswordBlur}
                required
              />
              {PasswordError && (
                <div className="error-message">{PasswordError}</div>
              )}

              <button onClick={LogInUser} className="signin-btn">
                Sign in
              </button>

              <div className="extra-buttons">
                <p className="or">&#x2015; Or &#x2015;</p>
                <button onClick={GoogleAuth} className="google">
                  <p>Sign in with</p>
                  <img src={google} className="google-img" alt="google" />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Signin;
