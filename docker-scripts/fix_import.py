#!/usr/bin/env python3
import os
import site
import sys
import shutil

def fix_imports():
    """
    Fix Python imports by creating necessary symlinks and adjusting paths
    """
    print("Running Python import fix utility...")
    
    # Get the application directory
    app_dir = "/app"
    nudenet_dir = os.path.join(app_dir, "nudenet")
    
    # Check if nudenet directory exists
    if not os.path.exists(nudenet_dir):
        print(f"❌ ERROR: nudenet directory not found at {nudenet_dir}")
        return False
    
    # Get site-packages directories
    site_packages = site.getsitepackages()
    dist_packages = [d for d in site_packages if "dist-packages" in d]
    
    if not dist_packages:
        print("❌ ERROR: Could not find dist-packages directory")
        return False
    
    # Create symlink in all site-packages directories
    success = False
    for site_dir in site_packages:
        target = os.path.join(site_dir, "nudenet")
        
        # Remove existing directory or symlink if present
        if os.path.exists(target):
            if os.path.islink(target):
                os.unlink(target)
                print(f"Removed existing symlink at {target}")
            elif os.path.isdir(target):
                shutil.rmtree(target)
                print(f"Removed existing directory at {target}")
        
        # Create symlink
        try:
            os.symlink(nudenet_dir, target)
            print(f"✅ Created symlink: {nudenet_dir} -> {target}")
            success = True
        except Exception as e:
            print(f"❌ Failed to create symlink at {target}: {e}")
    
    # Create .pth file to add app directory to Python path
    try:
        pth_file = os.path.join(dist_packages[0], "nudenet.pth")
        with open(pth_file, "w") as f:
            f.write(app_dir + "\n")
        print(f"✅ Created .pth file at {pth_file}")
        success = True
    except Exception as e:
        print(f"❌ Failed to create .pth file: {e}")
    
    # Test import
    try:
        sys.path.insert(0, app_dir)
        import nudenet
        print(f"✅ Successfully imported nudenet from {nudenet.__file__}")
        return True
    except ImportError as e:
        print(f"❌ Import still failing: {e}")
        return success

if __name__ == "__main__":
    if fix_imports():
        print("Import fix completed successfully")
        sys.exit(0)
    else:
        print("Import fix failed")
        sys.exit(1)