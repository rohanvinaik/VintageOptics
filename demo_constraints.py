"""
Demo script showing the new constraint-based orchestration system in action.

This demonstrates:
1. Constraint validation in lens analysis
2. Orthogonal error decomposition
3. Task graph execution
4. Uncertainty tracking
5. Semantic tool orchestration
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# Import new components
from vintageoptics.constraints import (
    ConstraintSpecification,
    OrthogonalErrorAnalyzer,
    OpticalTaskGraph,
    TaskNode,
    UncertaintyTracker
)
from vintageoptics.orchestration import (
    OpticalToolRegistry,
    SemanticOpticalOrchestrator
)
from vintageoptics.analysis import LensCharacterizer


def demo_constraint_validation():
    """Demonstrate physical constraint validation."""
    print("\n=== Constraint Validation Demo ===")
    
    # Create constraint specification
    constraints = ConstraintSpecification()
    
    # Create a test image
    image = np.random.rand(512, 512, 3).astype(np.float32)
    
    # Simulate a correction that violates energy conservation
    corrected = image * 1.5  # Artificially brighten
    
    # Add metadata
    metadata = {
        'f_stop': 2.8,
        'resolution': 512,
        'spherical_aberration': 2.5,
        'chromatic_aberration': 5.0
    }
    
    # Validate
    results = constraints.validate_correction(image, corrected, metadata)
    
    print("\nValidation Results:")
    for constraint, (valid, message) in results.items():
        status = "✓" if valid else "✗"
        print(f"  {status} {constraint}: {message or 'OK'}")
    
    # Export constraints for sharing
    constraints.export_constraints("lens_constraints.json")
    print("\nExported constraints to lens_constraints.json")


def demo_orthogonal_error_analysis():
    """Demonstrate orthogonal error decomposition."""
    print("\n=== Orthogonal Error Analysis Demo ===")
    
    # Create test image with mixed errors
    size = (256, 256, 3)
    clean_image = np.zeros(size, dtype=np.float32)
    
    # Add gradient (clean signal)
    for i in range(size[0]):
        clean_image[i, :] = i / size[0]
    
    # Add analog errors (smooth vignetting)
    y, x = np.ogrid[:size[0], :size[1]]
    center = (size[0]//2, size[1]//2)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2) / (size[0]//2)
    vignetting = 1 - 0.5 * r**2
    analog_error = clean_image * vignetting[:, :, np.newaxis]
    
    # Add digital errors (pixel noise)
    digital_error = analog_error + np.random.normal(0, 0.1, size)
    
    # Clip to valid range
    noisy_image = np.clip(digital_error, 0, 1)
    
    # Analyze errors
    analyzer = OrthogonalErrorAnalyzer()
    decomposition = analyzer.decompose_errors(
        noisy_image,
        lens_profile={'type': 'vintage'},
        sensor_profile={'type': 'cmos'}
    )
    
    print("\nError Decomposition:")
    print(f"  Analog errors found: {list(decomposition['analog_errors'].keys())}")
    print(f"  Digital errors found: {list(decomposition['digital_errors'].keys())}")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(noisy_image)
    axes[0, 0].set_title("Input (with errors)")
    
    axes[0, 1].imshow(decomposition['clean_signal'])
    axes[0, 1].set_title("Recovered Clean Signal")
    
    axes[0, 2].imshow(decomposition['error_map'], cmap='hot')
    axes[0, 2].set_title("Error Confidence Map")
    
    # Show specific error components
    if 'vignetting' in decomposition['analog_errors']:
        axes[1, 0].imshow(decomposition['analog_errors']['vignetting'], cmap='gray')
        axes[1, 0].set_title("Vignetting Component")
    
    if 'shot_noise' in decomposition['digital_errors']:
        axes[1, 1].imshow(decomposition['digital_errors']['shot_noise'], cmap='gray')
        axes[1, 1].set_title("Shot Noise Component")
    
    axes[1, 2].imshow(noisy_image - decomposition['clean_signal'])
    axes[1, 2].set_title("Total Error")
    
    plt.tight_layout()
    plt.savefig("error_decomposition_demo.png", dpi=150)
    print("\nSaved visualization to error_decomposition_demo.png")
    plt.close()


def demo_task_graph():
    """Demonstrate task graph construction and execution."""
    print("\n=== Task Graph Demo ===")
    
    # Create task graph
    graph = OpticalTaskGraph()
    
    # Define pipeline
    tasks = [
        TaskNode(
            id="analyze",
            name="Analyze Lens Characteristics",
            task_type="lens_analyzer",
            parameters={"full_analysis": True},
            outputs=["lens_profile", "defects"]
        ),
        TaskNode(
            id="correct",
            name="Correct Optical Errors",
            task_type="error_correction",
            parameters={"preserve_character": 0.3},
            inputs=["lens_profile"],
            outputs=["corrected_image"],
            constraints=["energy_conservation", "diffraction_limit"]
        ),
        TaskNode(
            id="enhance",
            name="Enhance Image Quality",
            task_type="enhancement",
            parameters={"sharpness": 0.7},
            inputs=["corrected_image"],
            outputs=["final_image"]
        )
    ]
    
    # Build graph
    for task in tasks:
        graph.add_task(task)
    
    graph.add_dependency("analyze", "correct", "lens_profile")
    graph.add_dependency("correct", "enhance", "corrected_image")
    
    # Validate
    is_valid, errors = graph.validate_graph()
    print(f"\nGraph validation: {'✓ Valid' if is_valid else '✗ Invalid'}")
    if errors:
        for error in errors:
            print(f"  - {error}")
    
    # Visualize
    dot_graph = graph.visualize("pipeline_graph.dot")
    print("\nGenerated graph visualization (DOT format):")
    print(dot_graph[:200] + "..." if len(dot_graph) > 200 else dot_graph)
    
    # Export specification
    spec = graph.export_spec()
    print(f"\nPipeline specification:")
    print(f"  Tasks: {len(spec['tasks'])}")
    print(f"  Dependencies: {len(spec['dependencies'])}")
    
    # Save for reuse
    graph.save("lens_correction_pipeline.json")
    print("\nSaved pipeline to lens_correction_pipeline.json")


def demo_uncertainty_tracking():
    """Demonstrate uncertainty quantification."""
    print("\n=== Uncertainty Tracking Demo ===")
    
    # Create test image
    image = np.random.rand(256, 256, 3).astype(np.float32)
    
    # Create tracker
    tracker = UncertaintyTracker()
    
    # Estimate input uncertainty
    metadata = {
        'iso': 3200,  # High ISO = more noise
        'bit_depth': 8
    }
    
    input_uncertainty = tracker.estimate_input_uncertainty(image, metadata)
    print(f"\nInput uncertainty (mean): {np.mean(input_uncertainty.std):.3f}")
    print(f"Input uncertainty (max): {np.max(input_uncertainty.std):.3f}")
    
    # Define processing pipeline
    pipeline = [
        ('blur', {'kernel_size': 5}),
        ('color_correction', {'color_matrix': np.eye(3) * 1.1}),
        ('distortion_correction', {'model_order': 3}),
        ('sharpening', {'strength': 0.5})
    ]
    
    # Track through pipeline
    final_uncertainty = tracker.propagate_through_pipeline(
        input_uncertainty, pipeline
    )
    
    print(f"\nFinal uncertainty (mean): {np.mean(final_uncertainty.std):.3f}")
    print(f"Final uncertainty (max): {np.max(final_uncertainty.std):.3f}")
    print(f"Uncertainty increase: {np.mean(final_uncertainty.std) / np.mean(input_uncertainty.std):.1f}x")
    
    # Visualize uncertainty
    viz = tracker.visualize_uncertainty(image, final_uncertainty, "uncertainty_map.png")
    print("\nSaved uncertainty visualization to uncertainty_map.png")


def demo_semantic_orchestration():
    """Demonstrate natural language pipeline construction."""
    print("\n=== Semantic Orchestration Demo ===")
    
    # Create registry and orchestrator
    registry = OpticalToolRegistry()
    orchestrator = SemanticOpticalOrchestrator(registry)
    
    # Test various natural language requests
    requests = [
        "Make this photo look like it was shot on a vintage Helios lens",
        "Apply professional correction to fix all optical issues",
        "Create an artistic portrait with beautiful bokeh",
        "Fix chromatic aberration but preserve the vintage character"
    ]
    
    for request in requests:
        print(f"\nRequest: '{request}'")
        
        # Process request
        result = orchestrator.process_request(request)
        
        print(f"Intent: {result['intent']}")
        print(f"Entities: {result['entities']}")
        print("\nExecution plan:")
        print(result['explanation'])
        
        # Generate code
        code = orchestrator.generate_code(result)
        print("\nGenerated code:")
        print(code[:300] + "..." if len(code) > 300 else code)
        
        # Get alternatives
        alternatives = orchestrator.suggest_alternatives(request, result)
        if alternatives:
            print("\nAlternative approaches:")
            for alt in alternatives[:2]:
                print(f"  - {alt['description']}: {alt['reason']}")


def demo_integrated_analysis():
    """Demonstrate integrated lens analysis with all new features."""
    print("\n=== Integrated Analysis Demo ===")
    
    # Create test image with known characteristics
    size = (512, 512, 3)
    image = np.ones(size, dtype=np.uint8) * 128
    
    # Add some features
    cv2.circle(image, (256, 256), 100, (255, 255, 255), -1)
    cv2.rectangle(image, (100, 100), (200, 200), (64, 64, 64), -1)
    
    # Add simulated lens defects
    # Dust spots
    for _ in range(5):
        x, y = np.random.randint(50, 450, 2)
        cv2.circle(image, (x, y), 3, (0, 0, 0), -1)
    
    # Initialize enhanced characterizer
    characterizer = LensCharacterizer(
        use_hd=True,
        use_constraints=True,
        use_uncertainty=True
    )
    
    # Analyze with all features
    print("\nAnalyzing image with full feature set...")
    characteristics = characterizer.analyze(
        image,
        full_analysis=True,
        validate_constraints=True,
        track_uncertainty=True
    )
    
    # Generate comprehensive report
    report = characterizer.generate_report(characteristics)
    print("\n" + report)
    
    # Save report
    with open("integrated_analysis_report.txt", "w") as f:
        f.write(report)
    print("\nFull report saved to integrated_analysis_report.txt")


def demo_tool_registry():
    """Demonstrate the optical tool registry."""
    print("\n=== Tool Registry Demo ===")
    
    registry = OpticalToolRegistry()
    
    # Show registry statistics
    stats = registry.get_statistics()
    print("\nRegistry Statistics:")
    print(f"  Total tools: {stats['total_tools']}")
    print(f"  Tools by type: {stats['tools_by_type']}")
    print(f"  Popular tags: {stats['popular_tags'][:3]}")
    
    # Search for tools
    print("\nSearching for 'bokeh' tools:")
    bokeh_tools = registry.search_tools("bokeh", limit=3)
    for tool in bokeh_tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Find compatible tools
    print("\nFinding tools that can process 'image' input:")
    image_tools = registry.find_tools_by_capability(input_type="image")
    print(f"  Found {len(image_tools)} compatible tools")
    
    # Get tool chain suggestion
    print("\nTool chain suggestion for 'vintage portrait':")
    chain = registry.get_tool_chain_suggestion("vintage portrait")
    for i, tool_id in enumerate(chain, 1):
        tool = registry.get_tool(tool_id)
        if tool:
            print(f"  {i}. {tool.name}")
    
    # Export registry
    registry.export_registry("optical_tools_registry.json")
    print("\nExported registry to optical_tools_registry.json")


def main():
    """Run all demonstrations."""
    print("=== VintageOptics Constraint-Based Orchestration Demo ===")
    print("This demonstrates the new MVP components for enhanced lens processing")
    
    demos = [
        ("Constraint Validation", demo_constraint_validation),
        ("Orthogonal Error Analysis", demo_orthogonal_error_analysis),
        ("Task Graph Construction", demo_task_graph),
        ("Uncertainty Tracking", demo_uncertainty_tracking),
        ("Tool Registry", demo_tool_registry),
        ("Semantic Orchestration", demo_semantic_orchestration),
        ("Integrated Analysis", demo_integrated_analysis)
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\nError in {name} demo: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== Demo Complete ===")
    print("Generated files:")
    print("  - lens_constraints.json")
    print("  - error_decomposition_demo.png")
    print("  - pipeline_graph.dot")
    print("  - uncertainty_map.png")
    print("  - lens_correction_pipeline.json")
    print("  - optical_tools_registry.json")
    print("  - integrated_analysis_report.txt")


if __name__ == "__main__":
    main()
