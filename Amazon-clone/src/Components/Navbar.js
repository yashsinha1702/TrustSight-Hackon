import { React, useEffect, useState, useRef } from "react";
import Logo from "../imgs/logo.png";
import LogoSmall from "../imgs/A-logo.png";
import search from "../imgs/search.png";
import wishlist from "../imgs/wishlist.png";
import cart from "../imgs/cart.png";
import orders from "../imgs/orders.png";
import Default from "../imgs/default.png";
import { useSelector } from "react-redux";
import { useNavigate } from "react-router-dom";
import "./navbar.css";
import { app } from "../Firebase";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import swal from "sweetalert";

const auth = getAuth(app);

function Navbar() {
  const CartItems = useSelector((state) => state.CartItemsAdded.CartItems);
  const ListItems = useSelector((state) => state.ItemsAdded.ListItems);
  const OrderItems = useSelector((state) => state.OrderAdded.OrderItems);
  const [user, setUser] = useState(null);
  const [searchText, setSearchText] = useState("");
  const [Products, setProducts] = useState([]);
  const [isAdmin, setIsAdmin] = useState(false);

  const navigate = useNavigate();

  const searchResultsRef = useRef(null);

  const totalLength = OrderItems.reduce((acc, item) => {
    if (Array.isArray(item)) {
      return acc + item.length;
    }
    return acc + 1;
  }, 0);

  useEffect(() => {
    onAuthStateChanged(auth, (user) => {
      if (user) {
        setUser(user);
        // Check if user is admin
        setIsAdmin(user.email && (user.email.includes('admin') || user.email.includes('demo')));
      } else {
        setUser(null);
        setIsAdmin(false);
      }
    });

    const GetProducts = async () => {
      const data = await fetch("https://fakestoreapi.com/products");
      const new_data = await data.json();
      setProducts(new_data);
    };

    GetProducts();

    const handleClick = (event) => {
      if (
        searchResultsRef.current &&
        !searchResultsRef.current.contains(event.target)
      ) {
        setSearchText("");
      }
    };
    document.addEventListener("click", handleClick);
    return () => {
      document.removeEventListener("click", handleClick);
    };
  }, []);

  const searchResults = Products.filter(
    (product) =>
      product.title.toLowerCase().includes(searchText.toLowerCase()) ||
      product.description.toLowerCase().includes(searchText.toLowerCase())
  );

  const totalQuantity = CartItems.reduce(
    (total, item) => total + item.quantity,
    0
  );

  return (
    <>
      <div className="navbar">
        <div className="left-section">
          <img
            onClick={() => {
              if (window.location.href.includes("/payment")) {
                swal({
                  title: "Are you sure?",
                  text: "Your transaction is still pending!",
                  icon: "warning",
                  buttons: true,
                  dangerMode: true,
                }).then((willDelete) => {
                  if (willDelete) {
                    navigate("/");
                  }
                });
              } else {
                navigate("/");
              }
            }}
            className="logo"
            src={Logo}
            alt="logo"
          />
          <img className="logo-small" src={LogoSmall} alt="logo" />
        </div>

        <div className="middle-section">
          <div className="search-bar">
            <input
              type="text"
              placeholder="Search amazon"
              className="search-input"
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
            />
            <img src={search} alt="search" className="search-icon" />
          </div>
          {searchText && (
            <div className="search-results" ref={searchResultsRef}>
              {searchResults.map((product) => (
                <div
                  key={product.id}
                  className="search-result-item"
                  onClick={() => {
                    navigate(`/product/${product.id}`);
                    setSearchText("");
                  }}
                >
                  <img
                    src={product.image}
                    alt={product.title}
                    className="search-result-image"
                  />
                  <div className="search-result-details">
                    <p className="search-result-title">{product.title}</p>
                    <p className="search-result-price">${product.price}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="right-section">
          {/* TrustSight Dashboard Link - Only for Admin */}
          {user && isAdmin && (
            <div 
              className="trustsight-nav-item"
              onClick={() => navigate("/trustsight")}
            >
              <div className="trustsight-icon">🛡️</div>
              <p className="nav-text">TrustSight</p>
            </div>
          )}

          <div
            onClick={() => {
              if (window.location.href.includes("/payment")) {
                swal({
                  title: "Are you sure?",
                  text: "Your transaction is still pending!",
                  icon: "warning",
                  buttons: true,
                  dangerMode: true,
                }).then((willDelete) => {
                  if (willDelete) {
                    navigate("/wishlist");
                  }
                });
              } else {
                navigate("/wishlist");
              }
            }}
            className="wishlist-container"
          >
            <img src={wishlist} alt="wishlist" className="wishlist-icon" />
            <p className="nav-text">Wishlist</p>
            <div className="wishlist-count">{ListItems.length}</div>
          </div>

          <div
            onClick={() => {
              if (window.location.href.includes("/payment")) {
                swal({
                  title: "Are you sure?",
                  text: "Your transaction is still pending!",
                  icon: "warning",
                  buttons: true,
                  dangerMode: true,
                }).then((willDelete) => {
                  if (willDelete) {
                    navigate("/cart");
                  }
                });
              } else {
                navigate("/cart");
              }
            }}
            className="cart-container"
          >
            <img src={cart} alt="cart" className="cart-icon" />
            <p className="nav-text">Cart</p>
            <div className="cart-count">{totalQuantity}</div>
          </div>

          <div
            onClick={() => {
              if (window.location.href.includes("/payment")) {
                swal({
                  title: "Are you sure?",
                  text: "Your transaction is still pending!",
                  icon: "warning",
                  buttons: true,
                  dangerMode: true,
                }).then((willDelete) => {
                  if (willDelete) {
                    navigate("/orders");
                  }
                });
              } else {
                navigate("/orders");
              }
            }}
            className="orders-container"
          >
            <img src={orders} alt="orders" className="orders-icon" />
            <p className="nav-text">Orders</p>
            <div className="orders-count">{totalLength}</div>
          </div>

          <div
            onClick={() => {
              if (window.location.href.includes("/payment")) {
                swal({
                  title: "Are you sure?",
                  text: "Your transaction is still pending!",
                  icon: "warning",
                  buttons: true,
                  dangerMode: true,
                }).then((willDelete) => {
                  if (willDelete) {
                    navigate("/profile");
                  }
                });
              } else {
                navigate("/profile");
              }
            }}
            className="profile-container"
          >
            <img
              src={user?.photoURL || Default}
              alt="profile"
              className="profile-pic"
            />
            <p className="nav-text">{user?.displayName || "Profile"}</p>
          </div>
        </div>
      </div>
    </>
  );
}

export default Navbar;