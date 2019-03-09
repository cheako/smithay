#!/bin/sh

set -ex

case $FEATURES in
    "all")
        vagga cargo-"$TRAVIS_RUST_VERSION" test --all-features &&
        vagga cargo-"$TRAVIS_RUST_VERSION" doc --no-deps --all-features
    ;;
    "default")
        vagga cargo-"$TRAVIS_RUST_VERSION" test &&
        vagga cargo-"$TRAVIS_RUST_VERSION" doc --no-deps
    ;;
    "examples")
        vagga cargo-"$TRAVIS_RUST_VERSION" check --examples
    ;;
    "cargo-fmt")
        vagga cargo-"$TRAVIS_RUST_VERSION" fmt --all -- --check
    ;;
    "cargo-clippy")
        vagga cargo-"$TRAVIS_RUST_VERSION" clippy --all-features -- -D warnings
    ;;
    "anvil")
        cd anvil
        case $ANVIL_FEATURES in
            "all")
                vagga cargo-"$TRAVIS_RUST_VERSION" test --all-features
            ;;
            "default")
                vagga cargo-"$TRAVIS_RUST_VERSION" test
            ;;
            *)
                vagga cargo-"$TRAVIS_RUST_VERSION" check --no-default-features --features "$ANVIL_FEATURES"
        esac
    ;;
    "vkwayland")
        cd vkwayland
        case $VKWAYLAND_FEATURES in
            "all")
                vagga cargo-"$TRAVIS_RUST_VERSION" test --all-features
            ;;
            "default")
                vagga cargo-"$TRAVIS_RUST_VERSION" test
            ;;
            *)
                vagga cargo-"$TRAVIS_RUST_VERSION" check --no-default-features --features "$VKWAYLAND_FEATURES"
        esac
    ;;
    *)
        vagga cargo-"$TRAVIS_RUST_VERSION" check --tests --no-default-features --features "$FEATURES" &&
        vagga cargo-"$TRAVIS_RUST_VERSION" doc --no-deps --no-default-features --features "$FEATURES"
esac
