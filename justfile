# Just file: https://github.com/casey/just
#
test:
	pytest

release version: test
    git tag {{version}}
    git checkout {{version}}
    git push --tags -f
    python3 setup.py sdist
    twine upload dist/conebeam_projector-{{version}}.tar.gz
    git checkout master
    
