from flask import Flask, render_template, Response, jsonify, send_from_directory, request, redirect, url_for, flash, session
import cv2
import numpy as np
import time
import os
from detect import detect_objects
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from authlib.integrations.flask_client import OAuth
from datetime import datetime
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['SECURITY_PASSWORD_SALT'] = 'my_precious_two'
s = URLSafeTimedSerializer(app.config['SECRET_KEY'])
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Email Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'aryankumar735588@gmail.com'  # Your email
app.config['MAIL_PASSWORD'] = 'your-app-password-here'      # REPLACE THIS WITH YOUR APP PASSWORD
app.config['MAIL_DEFAULT_SENDER'] = 'aryankumar735588@gmail.com'

mail = Mail(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

oauth = OAuth(app)

# Google & Facebook Configuration (Replace with your actual keys)
app.config['GOOGLE_CLIENT_ID'] = os.environ.get('GOOGLE_CLIENT_ID', 'your-google-client-id')
app.config['GOOGLE_CLIENT_SECRET'] = os.environ.get('GOOGLE_CLIENT_SECRET', 'your-google-client-secret')
app.config['FACEBOOK_CLIENT_ID'] = os.environ.get('FACEBOOK_CLIENT_ID', 'your-facebook-client-id')
app.config['FACEBOOK_CLIENT_SECRET'] = os.environ.get('FACEBOOK_CLIENT_SECRET', 'your-facebook-client-secret')

google = oauth.register(
    name='google',
    client_id=app.config['GOOGLE_CLIENT_ID'],
    client_secret=app.config['GOOGLE_CLIENT_SECRET'],
    access_token_url='https://accounts.google.com/o/oauth2/token',
    access_token_params=None,
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params=None,
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    client_kwargs={'scope': 'openid email profile'},
)

facebook = oauth.register(
    name='facebook',
    client_id=app.config['FACEBOOK_CLIENT_ID'],
    client_secret=app.config['FACEBOOK_CLIENT_SECRET'],
    access_token_url='https://graph.facebook.com/oauth/access_token',
    access_token_params=None,
    authorize_url='https://www.facebook.com/dialog/oauth',
    authorize_params=None,
    api_base_url='https://graph.facebook.com/',
    client_kwargs={'scope': 'email'},
)

# MODELS
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=True) # Nullable for OAuth users
    is_admin = db.Column(db.Boolean, default=False)
    oauth_provider = db.Column(db.String(50), nullable=True)

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('feedbacks', lazy=True, cascade="all, delete-orphan"))

class TeamMember(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(100), nullable=False)
    image_url = db.Column(db.String(500), nullable=False) # URL to image

@login_manager.user_loader
def load_user(user_id):
    user = User.query.get(int(user_id))
    # Robust check: Ensure specific email is ALWAYS admin (checking both correct and typoed versions)
    admin_emails = ['aryankumar735588@gmail.com', 'aryankumar735588@gmial.com']
    
    if user and user.email.lower().strip() in admin_emails and not user.is_admin:
        print(f"DEBUG: Force promoting {user.email} to Admin via load_user")
        user.is_admin = True
        db.session.commit()
    return user

# Initialize DB
with app.app_context():
    db.create_all()
    # Create default admin if not exists
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', email='admin@example.com', password=generate_password_hash('admin123'), is_admin=True)
        db.session.add(admin)
        db.session.commit()


# Camera is opened lazily and re-opened when needed so app starts even without a camera
camera = None
# When VIDEO_SOURCE env var is set to 'sample', generate an animated test stream (no physical camera needed)
# Default to camera 0 (real camera) instead of sample mode
src_env = os.getenv('VIDEO_SOURCE', '0')
sample_mode = isinstance(src_env, str) and src_env.strip().lower() == 'sample'
if sample_mode:
    print("Using synthetic sample video stream (no camera required)")
else:
    print("Using real camera (default camera 0)")
detecting = True

# Global stats tracking
current_stats = {
    "faces": 0,
    "humans": 0,
    "vehicles": 0,
    "cars": 0,
    "motorcycles": 0,
    "buses": 0,
    "trucks": 0,
    "traffic_lights": 0,
    "dogs": 0,
    "cats": 0,
    "cows": 0,
    "horses": 0,
    "zebra_crossings": 0,
    "footpaths": 0,
    "buffaloes": 0,
    "bullock_carts": 0,
    "fps": 0
}


def get_camera():
    """Return a working cv2.VideoCapture or None if it cannot be opened."""
    global camera
    if camera is None or not getattr(camera, 'isOpened', lambda: False)():
        # allow VIDEO_SOURCE env var to specify device index, file path or URL
        src = os.getenv('VIDEO_SOURCE', '0')
        global sample_mode
        try:
            # special value 'sample' uses a generated animated stream for testing without hardware
            if isinstance(src, str) and src.strip().lower() == 'sample':
                sample_mode = True
                print("Using synthetic sample video stream (no camera required)")
                camera = None
                return camera

            if isinstance(src, str) and src.isdigit():
                src_val = int(src)
            else:
                src_val = src

            print("Opening camera source:", src_val)
            # Try opening with different backends for better compatibility
            camera = cv2.VideoCapture(src_val, cv2.CAP_DSHOW)  # Use DirectShow on Windows
            if not camera.isOpened():
                camera = cv2.VideoCapture(src_val)  # Fallback to default
            
            time.sleep(0.5)

            if not camera.isOpened():
                print("Camera failed to open")
                try:
                    camera.release()
                except Exception:
                    pass
                camera = None
            else:
                print("Camera opened successfully")
                # Set some properties for better performance
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        except Exception as e:
            print("Camera exception:", e)
            camera = None

    return camera


# ========================= ROUTES ========================= #

@app.route('/team')
def team():
    return render_template('team.html')

@app.route("/")
def index():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    
    # Fetch team members
    try:
        team_members = TeamMember.query.all()
    except:
        team_members = []
        
    # Main page (Dashboard will be default section)
    return render_template("index.html", team_members=team_members)


@app.route("/toggle")
def toggle():
    global detecting
    detecting = not detecting
    return jsonify({"status": detecting})


@app.route("/status")
def status():
    """Return current detection status for the UI to poll safely."""
    return jsonify({"status": detecting})


@app.route("/stats")
def stats():
    """Return current detection statistics."""
    global current_stats
    return jsonify(current_stats)


@app.route("/dashboard")
def dashboard():
    return render_template("index.html", section="dashboard")


@app.route("/about")
def about():
    return render_template("index.html", section="about")

# ========================= AUTH ROUTES ========================= #

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and user.password and check_password_hash(user.password, password):
            # Auto-promote specific admin email if not already admin
            # Case-insensitive check and strip whitespace
            admin_emails = ['aryankumar735588@gmail.com', 'aryankumar735588@gmial.com']
            if user.email.strip().lower() in admin_emails and not user.is_admin:
                print(f"Promoting {user.email} to Admin...")
                user.is_admin = True
                db.session.commit()
            
            login_user(user)
            global detecting
            detecting = True
            return redirect(url_for('index'))
        else:
            flash('Login failed. Check your email and password.', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already exists.', 'warning')
            return redirect(url_for('signup'))
        
        # Check if this is the specific admin email
        is_admin_user = (email == 'aryankumar735588@gmail.com')
        
        new_user = User(username=username, email=email, password=generate_password_hash(password), is_admin=is_admin_user)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        global detecting
        detecting = True
        return redirect(url_for('index'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    global detecting, camera
    detecting = False
    
    # Explicitly release camera
    if camera is not None:
        try:
            camera.release()
            print("Camera released on logout")
        except Exception as e:
            print(f"Error releasing camera: {e}")
        camera = None

    logout_user()
    return redirect(url_for('login'))

@app.route('/login/google')
def login_google():
    redirect_uri = url_for('authorize_google', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/authorize/google')
def authorize_google():
    token = google.authorize_access_token()
    user_info = google.get('userinfo').json()
    email = user_info['email']
    user = User.query.filter_by(email=email).first()
    if not user:
        is_admin_user = (email == 'aryankumar735588@gmail.com')
        user = User(username=user_info['name'], email=email, oauth_provider='google', is_admin=is_admin_user)
        db.session.add(user)
        db.session.commit()
    login_user(user)
    global detecting
    detecting = True
    return redirect(url_for('index'))

@app.route('/login/facebook')
def login_facebook():
    redirect_uri = url_for('authorize_facebook', _external=True)
    return oauth.facebook.authorize_redirect(redirect_uri)

@app.route('/authorize/facebook')
def authorize_facebook():
    token = oauth.facebook.authorize_access_token()
    user_info = oauth.facebook.get('https://graph.facebook.com/me?fields=id,name,email').json()
    email = user_info.get('email')
    # fallback if email is not provided
    if not email:
        email = f"{user_info['id']}@facebook.com"
    
    user = User.query.filter_by(email=email).first()
    if not user:
        is_admin_user = (email == 'aryankumar735588@gmail.com')
        user = User(username=user_info['name'], email=email, oauth_provider='facebook', is_admin=is_admin_user)
        db.session.add(user)
        db.session.commit()
    login_user(user)
    global detecting
    detecting = True
    return redirect(url_for('index'))

@app.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        flash("Access Denied", "danger")
        return redirect(url_for('index'))
    users = User.query.all()
    feedbacks = Feedback.query.all()
    team_members = TeamMember.query.all()
    return render_template('admin.html', users=users, feedbacks=feedbacks, team_members=team_members)

@app.route('/admin/add_member', methods=['POST'])
@login_required
def add_team_member():
    if not current_user.is_admin:
        return redirect(url_for('index'))
    
    name = request.form.get('name')
    role = request.form.get('role')
    image_url = request.form.get('image_url') or 'https://ui-avatars.com/api/?name=' + name
    
    if name and role:
        member = TeamMember(name=name, role=role, image_url=image_url)
        db.session.add(member)
        db.session.commit()
        flash('Team member added successfully', 'success')
        
    return redirect(url_for('admin'))

@app.route('/admin/delete_member/<int:id>')
@login_required
def delete_team_member(id):
    if not current_user.is_admin:
        return redirect(url_for('index'))
        
    member = TeamMember.query.get_or_404(id)
    db.session.delete(member)
    db.session.commit()
    flash('Team member removed', 'info')
    return redirect(url_for('admin'))

@app.route('/admin/delete_user/<int:id>')
@login_required
def delete_user(id):
    if not current_user.is_admin:
        flash("Unauthorized access attempt recorded.", "danger")
        return redirect(url_for('index'))
    
    if current_user.id == id:
        flash("Critical Error: Cannot terminate current admin session.", "danger")
        return redirect(url_for('admin'))
        
    user = User.query.get_or_404(id)
    # Check if we are trying to delete the main admin email
    admin_emails = ['aryankumar735588@gmail.com', 'aryankumar735588@gmial.com']
    if user.email.lower() in admin_emails:
        flash("Access Denied: Cannot delete primary system administrator.", "danger")
        return redirect(url_for('admin'))

    try:
        db.session.delete(user)
        db.session.commit()
        flash(f"User '{user.username}' has been successfully terminated from the system.", "success")
    except Exception as e:
        db.session.rollback()
        print(f"Delete Error: {e}")
        flash(f"Error terminating user: {str(e)}", "danger")
        
    return redirect(url_for('admin'))

@app.route('/feedback', methods=['POST'])
@login_required
def submit_feedback():
    message = request.form.get('message')
    if message:
        fb = Feedback(user_id=current_user.id, message=message)
        db.session.add(fb)
        db.session.commit()
        
        # Send Email
        try:
            msg = Message(subject="New Feedback - Vision Assistant",
                          recipients=['aryankumar735588@gmail.com'])
            msg.body = f"User: {current_user.username} ({current_user.email})\n\nMessage:\n{message}"
            mail.send(msg)
            flash('Feedback sent! Email dispatched to admin.', 'success')
        except Exception as e:
            print("Email Error:", e)
            flash('Feedback saved, but email handling requires configuration.', 'warning')
            
    return redirect(url_for('index', section='feedback'))

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            token = s.dumps(email, salt=app.config['SECURITY_PASSWORD_SALT'])
            link = url_for('reset_password', token=token, _external=True)
            
            try:
                msg = Message('Password Reset Request', recipients=[email])
                msg.body = f'Your link to reset your password is {link}'
                mail.send(msg)
                flash('An email has been sent with instructions to reset your password.', 'info')
            except Exception as e:
                print("Email Reset Error:", e)
                # For development/demo purposes, we can show the link if sending fails
                flash(f'Password reset link generated (Email sending failed): {link}', 'warning')
            
            return redirect(url_for('login'))
        else:
            flash('Email address not found.', 'danger')
    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = s.loads(token, salt=app.config['SECURITY_PASSWORD_SALT'], max_age=3600)
    except:
        flash('The reset link is invalid or has expired.', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('reset_password.html', token=token)

        user = User.query.filter_by(email=email).first()
        if user:
            user.password = generate_password_hash(password)
            db.session.commit()
            flash('Your password has been updated!', 'success')
            return redirect(url_for('login'))
        else:
            flash('User not found.', 'danger')
            return redirect(url_for('login'))

    return render_template('reset_password.html', token=token)


# ========================= VIDEO STREAM ========================= #

def generate_frames():
    global current_stats
    reopen_counter = 0

    # Simple state for sample animation
    anim_pos = 0

    while True:
        # Synthetic sample stream mode (no camera required)
        if sample_mode:
            h, w = 480, 640
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            # moving rectangle to show motion
            x = anim_pos % (w - 120)
            cv2.rectangle(frame, (x + 20, 120), (x + 120, 220), (0, 200, 0), -1)
            cv2.putText(frame, "Sample Stream", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

            anim_pos += 8

            if detecting:
                try:
                    frame, fps, counts = detect_objects(frame)
                    # Update global stats
                    current_stats["fps"] = fps
                    current_stats["faces"] = counts.get("Faces", 0)
                    current_stats["humans"] = counts.get("Humans", 0)
                    current_stats["vehicles"] = counts.get("Vehicles", 0)
                    current_stats["cars"] = counts.get("Cars", 0)
                    current_stats["motorcycles"] = counts.get("Motorcycles", 0)
                    current_stats["buses"] = counts.get("Buses", 0)
                    current_stats["trucks"] = counts.get("Trucks", 0)
                    current_stats["traffic_lights"] = counts.get("Traffic_Lights", 0)
                    current_stats["dogs"] = counts.get("Dogs", 0)
                    current_stats["cats"] = counts.get("Cats", 0)
                    current_stats["cows"] = counts.get("Cows", 0)
                    current_stats["horses"] = counts.get("Horses", 0)
                    current_stats["zebra_crossings"] = counts.get("Zebra_Crossings", 0)
                    current_stats["footpaths"] = counts.get("Footpaths", 0)
                    current_stats["buffaloes"] = counts.get("Buffaloes", 0)
                    current_stats["bullock_carts"] = counts.get("Bullock_Carts", 0)
                except Exception as e:
                    fps = 0
                    counts = {}
                    print("Detection error:", e)
                    current_stats["fps"] = 0
                    current_stats["faces"] = 0
                    current_stats["humans"] = 0
                    current_stats["vehicles"] = 0
                    current_stats["cars"] = 0
                    current_stats["motorcycles"] = 0
                    current_stats["buses"] = 0
                    current_stats["trucks"] = 0
                    current_stats["traffic_lights"] = 0
                    current_stats["dogs"] = 0
                    current_stats["cats"] = 0
                    current_stats["cows"] = 0
                    current_stats["horses"] = 0
                    current_stats["zebra_crossings"] = 0
                    current_stats["footpaths"] = 0
                    current_stats["buffaloes"] = 0
                    current_stats["bullock_carts"] = 0
            else:
                fps = 0
                counts = {}

                fps = 0
                counts = {}

            # Add FPS and counts overlay same as live stream
            cv2.putText(frame, f"FPS: {fps}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y = 40
            for obj, cnt in counts.items():
                cv2.putText(frame, f"{obj}: {cnt}", (10, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y += 25

            ok, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

            # throttle synthetic stream to reasonable rate
            time.sleep(0.03)
            continue
        
        # --- End Sample Mode ---

        # If user logs out, stop detection logic
        # We check context safely. Note: streaming runs in a separate thread/context usually,
        # but accessing current_user outside request context needs care.
        # Simple approach: rely on 'detecting' flag or just let it run but stop if global detecting is False AND no one watching?
        # Better: just check if we should be detecting.
        
        # For simplicity in this loop:
        if not detecting:  
             # Release camera to save resources
             if camera is not None:
                 camera.release()
                 camera = None
             
             # Sent a placeholder
             frame = np.zeros((480, 640, 3), dtype=np.uint8)
             cv2.putText(frame, "Detection Paused", (180, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
             _, buffer = cv2.imencode(".jpg", frame)
             yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
             time.sleep(1.0) # Sleep longer when paused
             continue

        cam = get_camera()
        success = False
        frame = None

        if cam is not None:
            try:
                success, frame = cam.read()
            except Exception:
                success = False

        if not success or frame is None:
            # Placeholder frame when camera is not available
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera not available", (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # periodically try to re-open camera
            reopen_counter += 1
            if reopen_counter >= 10:
                reopen_counter = 0
                try:
                    if camera is not None:
                        camera.release()
                except Exception:
                    pass

        if detecting:
            try:
                frame, fps, counts = detect_objects(frame)
                # Update global stats
                current_stats["fps"] = fps
                current_stats["faces"] = counts.get("Faces", 0)
                current_stats["humans"] = counts.get("Humans", 0)
                current_stats["vehicles"] = counts.get("Vehicles", 0)
                current_stats["cars"] = counts.get("Cars", 0)
                current_stats["motorcycles"] = counts.get("Motorcycles", 0)
                current_stats["buses"] = counts.get("Buses", 0)
                current_stats["trucks"] = counts.get("Trucks", 0)
                current_stats["traffic_lights"] = counts.get("Traffic_Lights", 0)
                current_stats["dogs"] = counts.get("Dogs", 0)
                current_stats["cats"] = counts.get("Cats", 0)
                current_stats["cows"] = counts.get("Cows", 0)
                current_stats["horses"] = counts.get("Horses", 0)
                current_stats["zebra_crossings"] = counts.get("Zebra_Crossings", 0)
                current_stats["footpaths"] = counts.get("Footpaths", 0)
                current_stats["buffaloes"] = counts.get("Buffaloes", 0)
                current_stats["bullock_carts"] = counts.get("Bullock_Carts", 0)
            except Exception as e:
                fps = 0
                counts = {}
                print("Detection error:", e)
                cv2.putText(frame, "Detection error", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                current_stats["fps"] = 0
                current_stats["faces"] = 0
                current_stats["humans"] = 0
                current_stats["vehicles"] = 0
                current_stats["cars"] = 0
                current_stats["motorcycles"] = 0
                current_stats["buses"] = 0
                current_stats["trucks"] = 0
                current_stats["traffic_lights"] = 0
                current_stats["dogs"] = 0
                current_stats["cats"] = 0
                current_stats["cows"] = 0
                current_stats["horses"] = 0
                current_stats["zebra_crossings"] = 0
                current_stats["footpaths"] = 0
                current_stats["buffaloes"] = 0
                current_stats["bullock_carts"] = 0

            # FPS
            cv2.putText(frame, f"FPS: {fps}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Object-wise count
            y = 40
            for obj, cnt in counts.items():
                cv2.putText(frame, f"{obj}: {cnt}",
                            (10, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y += 25

        # Encode frame
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Frame encode error", (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            _, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ========================= STATIC FILES ========================= #

@app.route('/style.css')
def style():
    return send_from_directory('templates', 'style.css')


@app.route('/script.js')
def script():
    return send_from_directory('templates', 'script.js')


# ========================= RUN ========================= #

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)